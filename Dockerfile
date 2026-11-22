# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the organizers labeled every commit in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# Match the exact packages used during official labeling for runtime test verification.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

# sqlglot at the pinned commit, precompiled to .pyc for fast runtime verification.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/sqlglot-src \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app/sqlglot-src --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && python -m compileall -q /app/sqlglot-src

# Modular mypyc + Cython build of sqlglot C extension groups under /app/sqlglot-c.
COPY blast_radius/__init__.py blast_radius/c_loader.py blast_radius/build_c.py /app/blast_radius/
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gcc libc6-dev \
    && pip install --no-cache-dir cython \
    && PYTHONPATH=/app python -m blast_radius.build_c \
    && pip uninstall -y -q cython \
    && apt-get purge -y --auto-remove -qq gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && chmod -R a+rX /app/sqlglot-c

# Precompile TPC baseline before copying map/ and the rest of blast_radius so rebuilds stay cached.
COPY blast_radius/precompile_tpc.py /app/blast_radius/
RUN mkdir -p /app/map \
    && PYTHONPATH=/app python -m blast_radius.precompile_tpc

COPY map/ /app/map/
COPY blast_radius/ /app/blast_radius/
# Record pure-Python import state & per-case/DB caches first, then calibrate C + pure-Py costs.
RUN PYTHONPATH=/app python -m blast_radius.import_state \
    && SKIP_INTEGRATION=0 PYTHONPATH=/app python -m blast_radius.precompile_cases \
    && PYTHONPATH=/app python -m blast_radius.precompile \
    && python -c "import gzip; open('/app/map/values.marshal', 'wb').write(gzip.decompress(open('/app/map/values.bin', 'rb').read()))" \
    && python -m compileall -q /app/blast_radius \
    && chmod -R a+rX /app/blast_radius /app/map /app/sqlglot-c

ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["python", "-m", "blast_radius"]
