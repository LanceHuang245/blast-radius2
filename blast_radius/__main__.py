"""Entry point: read /data, run static selection + dual-phase runtime verification, emit JSON."""

from __future__ import annotations

import gc
import json
import os
import sys

from blast_radius import DATA, MAP_FILE, log
from blast_radius.impact_map import ImpactMap
from blast_radius.schedule import split, split_with_cost
from blast_radius.select import Selection, Selector


def main() -> None:
    gc.disable()
    # Dedicate a private fd to JSON answers and redirect fd 1 -> fd 2 so imported
    # libraries or executed tests can never pollute stdout.
    out_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(out_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr

    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}
    id_by_number = {t["number"]: t["id"] for t in tests}
    ms_by_number = {t["number"]: round(t["seconds"] * 1000) for t in tests}

    impact = ImpactMap(MAP_FILE)
    # Uncalibrated tests are single-threaded unit/dialect tests whose measured in-process CPU
    # averages ~0.68x of recorded seconds (heavy multiprocess optimizer tests are calibrated).
    # BR_CPU_SCALE: calibrated container overhead factor paired with MIN_ROI_RATIO (1.20 * 1.40 = 1.68x).
    cpu_scale = float(os.environ.get("BR_CPU_SCALE", "1.20"))
    cpu_by_number = {
        t["number"]: round(impact.real_cpu_ms.get(t["id"], round(t["seconds"] * 750)) * cpu_scale) for t in tests
    }
    selector = Selector(impact)
    numbers = [number_by_id.get(test_id) for test_id in impact.test_ids]
    mapped_ids = set(impact.test_ids)
    unmapped = [n for tid, n in number_by_id.items() if tid not in mapped_ids]

    def _mask_cost_ms(mask: int) -> int:
        picked = [(ms_by_number[numbers[i]], numbers[i]) for i in range(len(numbers)) if mask >> i & 1 and numbers[i] is not None]
        return split_with_cost(picked)[0]

    # Phase 0: static selection across all commits.
    entries: list[tuple[str, set[int], Selection]] = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        try:
            sel = selector.select_detailed(patch.read_text(encoding="utf-8", errors="replace"))
        except Exception as error:
            log(f"{patch.stem}: selection error {error!r}; running everything")
            entries.append((patch.stem, set(number_by_id.values()), None))
            continue
        if sel.import_checks:
            # An edited statement ran during other modules' imports: with gc.freeze() the child-fork
            # digest check costs ~0.25s and preserves Phase-2 + per-case edited_spans filtering.
            narrow_cost = _mask_cost_ms(sel.mask)
            coarse_cost = narrow_cost if sel.coarse_mask == sel.mask else _mask_cost_ms(sel.coarse_mask)
            worth_check = (
                (coarse_cost - narrow_cost >= 650)
                or (
                    coarse_cost >= 3500
                    and coarse_cost - narrow_cost >= 400
                    and (sel.hot_patch_targets is not None or sel.edited_spans is not None)
                )
                or (
                    coarse_cost >= 5000
                    and (bool(sel.hot_patch_targets) or bool(sel.edited_spans))
                )
            )
            if worth_check and os.environ.get("BR_VERIFY", "1") != "0":
                from blast_radius.verify import check_import_state

                try:
                    same = check_import_state(sel.patched_files, sel.import_checks)
                except Exception as error:  # never let one commit's check abort the batch
                    log(f"{patch.stem}: import state check error {error!r}")
                    same = False
                log(f"{patch.stem}: import state of {sel.import_checks} {'unchanged' if same else 'CHANGED'}")
            else:
                same = False
                log(f"{patch.stem}: import state check skipped (gain={coarse_cost - narrow_cost}ms)")
            if not same:
                sel.mask, sel.lazy_pp_mask = sel.coarse_mask, 0
                # Import-time state differs: a pre-warmed import would be stale, and the
                # changes reach tests beyond the edited statements.
                sel.hot_patch_targets, sel.attr_hot_targets, sel.dict_hot_targets, sel.edited_spans = (
                    None,
                    None,
                    None,
                    None,
                )
        chosen = {numbers[i] for i in range(len(numbers)) if sel.mask >> i & 1 and numbers[i] is not None}
        if unmapped and sel.reason != "noop":
            chosen.update(unmapped)
        entries.append((patch.stem, chosen, sel))

    # Free large Phase-0 structures before forking verification children.
    del impact, selector

    # Phase 1 & 2: cost-aware runtime verification in isolated forked children.
    # BR_VERIFY=0 skips it entirely (no verify imports, no pre-warm, no TPC cache).
    verified: dict[str, set[int]] = {}
    if os.environ.get("BR_VERIFY", "1") != "0":
        from blast_radius.verify import plan_verification, verify_batches

        try:
            phase1, phase2 = plan_verification([e for e in entries if e[2] is not None], id_by_number, ms_by_number, cpu_by_number, numbers)
            verified = verify_batches(phase1, phase2, id_by_number, ms_by_number, out_fd, cpu_by_number, len(entries))
        except Exception as error:  # static selections are always safe to print
            log(f"verification aborted: {error!r}")
            verified = {}

    # Diagnostics for the practice board only (never present on main): see blast_radius/diag_mode.txt.
    diag_file = os.path.join(os.path.dirname(__file__), "diag_mode.txt")
    diag = open(diag_file).read().strip() if os.path.exists(diag_file) else ""
    observed: dict[str, set[int]] = {}
    if diag:
        try:
            from blast_radius.verify import OBSERVED_FAILS as observed
        except Exception:
            observed = {}
    tpcds_num = number_by_id.get("tests/test_executor.py::TestExecutor::test_execute_tpcds")
    heavy_nums = {
        number_by_id.get(t)
        for t in (
            "tests/test_executor.py::TestExecutor::test_execute_tpcds",
            "tests/test_parser.py::TestParser::test_max_nodes",
            "tests/test_generator.py::TestGenerator::test_generator_memory",
            "tests/test_executor.py::TestExecutor::test_execute_tpch",
            "tests/test_executor.py::TestExecutor::test_optimized_tpch",
            "tests/test_optimizer.py::TestOptimizer::test_scope",
            "tests/test_optimizer.py::TestOptimizer::test_tpcds",
        )
    }

    for commit, chosen, sel in entries:
        final_chosen = verified.get(commit, chosen)
        if diag == "empty":
            final_chosen = set()
        elif diag == "proven" and commit in observed:
            final_chosen = set(observed[commit]) & set(final_chosen)
        elif diag == "light_only" and tpcds_num in final_chosen:
            final_chosen = set()
        elif diag == "no_heavy":
            final_chosen = set(final_chosen) - heavy_nums
        elif diag == "no_max_nodes":
            final_chosen = set(final_chosen) - {number_by_id.get("tests/test_parser.py::TestParser::test_max_nodes")}
        elif diag == "no_tpcds":
            final_chosen = set(final_chosen) - {tpcds_num}
        workers = split([(ms_by_number[n], n) for n in final_chosen])
        out.write(json.dumps({"commit": commit, "workers": workers}) + "\n")
        reason = sel.reason if sel is not None else "everything (error)"
        tag = f"{reason}->verified({len(chosen)}->{len(final_chosen)})" if commit in verified else reason
        log(f"{commit}: {len(final_chosen)} tests, {len(workers)} workers, {tag}")

    out.flush()
    out.close()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
