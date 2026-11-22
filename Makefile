IMAGE ?= blast-radius-entry
# On Windows, python3 is a Store stub: make sample PYTHON="uv run --no-project python"
PYTHON ?= python3
SAMPLE := $(CURDIR)/sample
OUTPUT := $(CURDIR)/output.jsonl

# The flags the scorer runs your image with. The scorer is linux/amd64, so on
# an Apple Silicon Mac --platform makes your local run match it.
RUN_FLAGS := --rm --platform linux/amd64 --network none --cpus 2 \
	--memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
	--tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
	--security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
	--volume $(SAMPLE)/commits:/data/commits:ro \
	--volume $(SAMPLE)/tests.json:/data/tests.json:ro

.PHONY: build sample check map values

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

# Runs your image once on the sample commits under the scorer's limits and
# scores what it printed.
sample: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	$(PYTHON) score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT)

# The output's rules only: every line well formed and every commit answered.
# A wrong selection still passes here; make sample counts the misses.
check: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	$(PYTHON) score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT) > /dev/null && echo "format ok"

# Rebuilds map/impact.json.gz from sqlglot at the pinned commit: runs its whole
# suite once with tools/map_plugin.py, then folds the results with
# tools/build_map.py. Needs .cache/sqlglot (the pinned checkout). Takes minutes.
MAP_IMAGE := blast-radius-map
map:
	docker build --platform linux/amd64 -f tools/Dockerfile.map --tag $(MAP_IMAGE) .
	rm -rf .cache/mapout && mkdir -p .cache/mapout map
	docker run --rm --platform linux/amd64 -v $(CURDIR)/.cache/sqlglot:/src:ro \
		-v $(CURDIR)/tools:/tools:ro -v $(CURDIR)/.cache/mapout:/out -w /src \
		-e PYTHONPATH=/src:/tools -e MAP_OUT=/out $(MAP_IMAGE) \
		python -m pytest tests -p map_plugin -p no:cacheprovider -q -o addopts= || true
	docker run --rm --platform linux/amd64 -v $(CURDIR)/.cache/sqlglot:/src:ro \
		-v $(CURDIR)/tools:/tools:ro -v $(CURDIR)/.cache/mapout:/out \
		-v $(SAMPLE)/tests.json:/tests.json:ro -v $(CURDIR)/map:/map $(MAP_IMAGE) \
		python /tools/build_map.py /src /out /tests.json /map/impact.json.gz

# Rebuilds map/values.bin (condition-outcome map) from sqlglot at the pinned commit: runs its
# whole suite under tools/value_plugin.py in parallel shards, then folds the shards with
# tools/build_values.py. Needs .cache/sqlglot (the pinned checkout) and the map image.
values:
	docker build --platform linux/amd64 -f tools/Dockerfile.map --tag $(MAP_IMAGE) .
	rm -rf .cache/valout && mkdir -p .cache/valout
	docker run --rm --platform linux/amd64 -v $(CURDIR)/.cache/sqlglot:/src:ro \
		-v $(CURDIR)/tools:/tools:ro -v $(CURDIR):/repo:ro -v $(CURDIR)/.cache/valout:/out \
		-v $(SAMPLE)/tests.json:/tests.json:ro -w /src -e PYTHONPATH=/src:/tools -e MAP_OUT=/out \
		-e SKIP_INTEGRATION=0 -e SKIP_IDS=tests/test_parser.py::TestParser::test_max_nodes $(MAP_IMAGE) \
		sh -c 'python /tools/run_values.py 10 & python /tools/run_values_ids.py 8 x tests/test_parser.py tests/test_optimizer.py & VALUE_SHARD=maxnodes SKIP_IDS= timeout 180 python -m pytest tests/test_parser.py::TestParser::test_max_nodes -p value_plugin -p no:cacheprovider -q -o addopts= || true; wait'
	docker run --rm --platform linux/amd64 -v $(CURDIR):/w -w /w $(MAP_IMAGE) \
		python tools/build_values.py .cache/valout sample/tests.json map/values.bin
