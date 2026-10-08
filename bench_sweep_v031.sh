#!/bin/bash
# =============================================================================
# GLM-5.3-Flash-FP8 Performance Sweep — DP=8 + EP=8 (DEP), single node
# ISL/OSL: 96K/32K and 8K/1K | MC=128 (override with MC=<n>)
# Node: banff-ccs-aus-p20-29 | 8× MI300X (gfx942)
# Image: vllm/vllm-openai-rocm:nightly-ac68c3087 + ROCm/aiter PR#5979 patch
#
# Companion to bench_sweep_glm53_flash.sh (TP=8). Same ISL/OSL/NP ladders and
# same MC so the two result sets are directly comparable.
#
# Why DEP for this model: 94.9% of the 305.8 GiB of weights are routed experts,
# so EP shards the bulk while DP keeps the MLA KV cache per-rank instead of
# replicated across TP ranks.
#   TP8      : 9,881,580 KV tokens total     ->  75.39x concurrency @131072 tok
#   DP8+EP8  : 7,147,064 KV tokens PER RANK  ->  54.53x per rank = ~436x total
# =============================================================================
set -uo pipefail

CONTAINER="${CONTAINER:-glm53flash-v031}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"
MODEL="${MODEL:-glm-5-3-flash-fp8}"
TOKENIZER="${TOKENIZER:-/mnt/models/GLM-5.3-Flash}"
MC="${MC:-128}"
# NP ladders. Defaults match bench_sweep_glm53_flash.sh (TP=8) for comparability.
# Override to push past MC — a ladder whose max is below MC never exercises the
# concurrency cap, so e.g. MC=256 needs NP values >= 256 to mean anything.
NP_LONG="${NP_LONG:-1 8 11 16 32 64 128}"      # ISL=96K OSL=32K
NP_SHORT="${NP_SHORT:-1 8 16 32 64 128 256 512 1024}"  # ISL=8K OSL=1K
# Set HIGH_CONC=1 to add a high-concurrency profile that exploits DEP headroom
# (TP=8 cannot hold these; it tops out near 75 resident sequences).
HIGH_CONC="${HIGH_CONC:-0}"

RESULTS_DIR="${RESULTS_DIR:-${HOME}/bench_results_glm53flash_dep8_mc${MC}_$(date +%Y%m%d_%H%M)}"
mkdir -p "${RESULTS_DIR}"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"

echo "=== GLM-5.3-Flash-FP8 DP=8 EP=8 MC=${MC} Sweep ==="
echo "Container: ${CONTAINER} | Port: ${PORT}"
echo "Model:     ${MODEL}"
echo "Tokenizer: ${TOKENIZER}"
echo "Results:   ${RESULTS_DIR}"
echo ""

# --- container present? -------------------------------------------------------
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "ERROR: container '${CONTAINER}' is not running." >&2
    echo "Launch it first (see docker run block in the header of this repo's RUNBOOK §12)." >&2
    exit 1
fi

# --- wait for server ----------------------------------------------------------
echo "Checking server..."
until curl -s "${BASE_URL}/v1/models" > /dev/null 2>&1; do
    echo "  Waiting..."; sleep 10
done
echo "Server ready!"

# --- CONFIRM the server really is DP8+EP8 ------------------------------------
# Without this you can silently benchmark a TP=8 server and compare it to itself.
echo ""
echo "--- parallelism check ---"
DP_RANKS=$(docker exec "${CONTAINER}" bash -lc \
    'grep -ohE "Worker_DP[0-9]+_EP[0-9]+" /work/logs/*.log 2>/dev/null | sort -u | wc -l' 2>/dev/null || echo 0)
KV_LINE=$(docker exec "${CONTAINER}" bash -lc \
    'grep -ohE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" /work/logs/*.log 2>/dev/null | tail -1' 2>/dev/null || true)
if [ "${DP_RANKS:-0}" -ge 8 ]; then
    echo "  OK: ${DP_RANKS} distinct Worker_DP*_EP* ranks found (DP8+EP8 active)"
else
    echo "  WARNING: only '${DP_RANKS}' Worker_DP*_EP* ranks found in the server log."
    echo "  Results may NOT be from a DP8+EP8 server. Verify before comparing to TP8."
fi
[ -n "${KV_LINE}" ] && echo "  per-rank KV: ${KV_LINE}"
echo ""

# --- csv header ---------------------------------------------------------------
echo "isl,osl,mc,num_prompts,duration_s,completed,output_tok_s,total_tok_s,req_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms" > "${SUMMARY_CSV}"

run_bench() {
    local ISL=$1 OSL=$2 NP=$3
    local STEM="bench_isl${ISL}_osl${OSL}_mc${MC}_np${NP}"
    local LOG="${RESULTS_DIR}/${STEM}.log"
    local REQUEST_TIMEOUT
    if   [ "${ISL}" -ge 64000 ]; then REQUEST_TIMEOUT=7200
    elif [ "${ISL}" -ge 16000 ]; then REQUEST_TIMEOUT=3600
    else                               REQUEST_TIMEOUT=1800
    fi

    echo "=== ISL=${ISL} OSL=${OSL} MC=${MC} np=${NP} timeout=${REQUEST_TIMEOUT}s === $(date)"
    docker exec \
        -e AIOHTTP_CLIENT_TIMEOUT="${REQUEST_TIMEOUT}" \
        "${CONTAINER}" vllm bench serve \
        --model "${MODEL}" \
        --base-url "${BASE_URL}" \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len "${OSL}" \
        --random-range-ratio 0.0 \
        --num-prompts "${NP}" \
        --max-concurrency "${MC}" \
        --tokenizer "${TOKENIZER}" \
        --num-warmups 1 \
        --temperature 0 \
        --save-result \
        --result-dir /work/bench_json \
        --result-filename "${STEM}.json" \
        2>&1 | tee "${LOG}"

    echo ""
    echo "--- Results ISL=${ISL} OSL=${OSL} np=${NP} ---"
    grep -E "Successful|Failed|Output token|Total token|TPOT|TTFT|Peak concurrent" \
        "${LOG}" 2>/dev/null || true

    # append a machine-readable row from the saved JSON
    docker exec "${CONTAINER}" python3 -c "
import json,sys
try:
    d=json.load(open('/work/bench_json/${STEM}.json'))
except Exception:
    sys.exit(0)
g=lambda k: d.get(k,'')
print(','.join(str(x) for x in [
    ${ISL}, ${OSL}, ${MC}, ${NP},
    round(float(g('duration') or 0),2), g('completed'),
    round(float(g('output_throughput') or 0),2),
    round(float(g('total_token_throughput') or 0),2),
    round(float(g('request_throughput') or 0),4),
    round(float(g('mean_ttft_ms') or 0),1), round(float(g('median_ttft_ms') or 0),1),
    round(float(g('p99_ttft_ms') or 0),1),
    round(float(g('mean_tpot_ms') or 0),2), round(float(g('median_tpot_ms') or 0),2),
    round(float(g('p99_tpot_ms') or 0),2),
]))
" 2>/dev/null >> "${SUMMARY_CSV}" || true

    echo ""
    sleep 30
}

docker exec "${CONTAINER}" mkdir -p /work/bench_json 2>/dev/null || true

# 96K/32K sweep
echo "======== Profile: ISL=96K OSL=32K  (NP: ${NP_LONG}) ========"
for NP in ${NP_LONG}; do
    run_bench 96000 32000 "${NP}"
done

# 8K/1K sweep
echo "======== Profile: ISL=8K OSL=1K  (NP: ${NP_SHORT}) ========"
for NP in ${NP_SHORT}; do
    run_bench 8192 1000 "${NP}"
done

# Optional: concurrency levels TP=8 physically cannot hold at long context.
if [ "${HIGH_CONC}" = "1" ]; then
    echo "======== Profile: high concurrency (DEP headroom) ========"
    for HMC in 256 384; do
        MC="${HMC}"
        echo "--- MC=${MC} ---"
        run_bench 96000 32000 "$((HMC*2))"
        run_bench 8192 1000 "$((HMC*4))"
    done
fi

echo "=== All benchmarks complete ==="
echo "Results: ${RESULTS_DIR}"
echo ""
echo "--- summary.csv ---"
column -s, -t "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
echo ""
ls -la "${RESULTS_DIR}"
