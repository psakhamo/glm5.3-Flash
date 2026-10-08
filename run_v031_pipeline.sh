#!/bin/bash
# Durable driver for the GLM-5.3-Flash v0.31.0 recipe.
#
#   tmux new-session -d -s glm53_v031 'bash /home/psakhamo/GLM53_Flash_v031/run_v031_pipeline.sh'
#
# Stages, in order. Each is gated on the previous: a stage that fails stops the
# pipeline with the reason on screen rather than producing a later result that
# looks real but was measured on a broken stack.
#
#   1 build image (v0.31.0 + vllm#59412 + vllm#57134)
#   2 IR-1 gate: GPUs must be free before the server starts
#   3 launch DP8+EP8, wait for 8/8 ranks
#   4 functional: reasoning + tool calling
#   5 needle-in-a-haystack: confirms #59412 took effect on the NEW base
#   6 perf sweep: 96K/32K and 8K/1K, MC=128 and MC=256
#
# The accuracy suite is deliberately NOT run here: it needs the gate repo's
# interpreter and its own env, and it should run against a server that has
# already passed 4 and 5.
set -uo pipefail

D=/home/psakhamo/GLM53_Flash_v031
LOG=$D/logs/pipeline.log
mkdir -p "$D/logs" "$D/results"
exec > >(tee -a "$LOG") 2>&1

say() { echo; echo "================ $(date '+%F %T')  $*"; }
die() { echo; echo "!!!! ABORT: $*"; echo "!!!! pipeline stopped; nothing after this ran."; sleep infinity; }

# ---------------------------------------------------------------- 1. build
say "[1/6] build glm53flash-vllm-v031:latest"
cd "$D" || die "no $D"
docker build -f Dockerfile -t glm53flash-vllm-v031:latest . || die "image build failed"
docker images glm53flash-vllm-v031 --format '  built {{.Repository}}:{{.Tag}} {{.Size}}'

# ------------------------------------------------------- 2. IR-1 GPU gate
say "[2/6] IR-1: GPUs must be free"
docker stop glm53flash-patched glm53flash-v031 >/dev/null 2>&1
sleep 20
LEFT=$(pgrep -af 'vllm serve|VLLM::|Magpie' 2>/dev/null | grep -cv 'claude-\|pgrep' || true)
echo "  foreign serving procs: ${LEFT:-0}"
[ "${LEFT:-0}" -gt 0 ] && die "GPUs still occupied; refusing to benchmark over another workload"

docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(getent group video | cut -d: -f3)" \
  --entrypoint bash glm53flash-vllm-v031:latest -lc 'rocm-smi --showmeminfo vram --json' 2>/dev/null \
| python3 -c "
import json,sys
d=json.load(sys.stdin); w=0
for k,v in d.items():
    t=u=None
    for kk,vv in v.items():
        kl=kk.lower()
        if 'used' in kl: u=float(vv)
        elif 'total' in kl: t=float(vv)
    w=max(w,100*u/t)
print(f'  worst VRAM {w:.3f}%')
sys.exit(0 if w<1 else 1)
" || die "VRAM above 1%; a leftover allocation would corrupt the baseline"

# ------------------------------------------------------------- 3. launch
say "[3/6] launch DP8+EP8"
bash "$D/launch_container.sh" || die "launch script failed"
for i in $(seq 1 80); do
    curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "  ready after ~$((i*15))s"; break; }
    docker exec glm53flash-v031 bash -lc 'pgrep -f "vllm serve" >/dev/null' 2>/dev/null \
      || die "server process died during boot; see $D/logs/server.log"
    sleep 15
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || die "server never became ready"
docker exec glm53flash-v031 bash -lc '
L=/work/logs/server.log
echo "  startups: $(grep -c "Application startup complete" $L)/8"
echo "  faults  : $(grep -c "Memory access fault" $L)"
grep -ohE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency[^|]*" $L | sort -u | sed "s/^/  /"'

# --------------------------------------------------------- 4. functional
say "[4/6] functional: reasoning + tool calling"
cp /home/psakhamo/GLM53_Flash_fp8kv_pr57134/test_functional.py "$D/" 2>/dev/null
sed -i 's|MODEL = "zai-org/GLM-5.3-Flash"|MODEL = "glm-5-3-flash-fp8"|' "$D/test_functional.py"
python3 "$D/test_functional.py" 2>&1 | tee "$D/results/functional.log" | tail -12
# 6/7 is the known-good result: the 7th is a chat-template artifact
# (reasoning_content empty), present on every build including the nightly.
PASSED=$(grep -c '^\[PASS\]' "$D/results/functional.log" || echo 0)
echo "  passed: $PASSED/7"
[ "$PASSED" -lt 6 ] && die "functional regression: only $PASSED/7 passed (expected >=6)"

# -------------------------------------------------------------- 5. needle
say "[5/6] needle-in-a-haystack (validates #59412 on the v0.31.0 base)"
cp /home/psakhamo/GLM53_Flash_pr59412/needle_test.py "$D/" 2>/dev/null
sed -i 's|MODEL = "zai-org/GLM-5.3-Flash"|MODEL = "glm-5-3-flash-fp8"|' "$D/needle_test.py"
python3 "$D/needle_test.py" 2>&1 | tee "$D/results/needle.log" | tail -20
grep -q "All needles retrieved" "$D/results/needle.log" \
  || die "needle retrieval FAILED on v0.31.0 -- #59412 may not have taken effect. Do NOT trust long-context numbers from this build."

# ----------------------------------------------------------- 6. perf sweep
say "[6/6] perf sweep"
cp /home/psakhamo/GLM53_Flash/bench_sweep_glm53_flash_dep8.sh "$D/bench_sweep_v031.sh"
sed -i -e 's|CONTAINER:-glm53flash-dep8|CONTAINER:-glm53flash-v031|' \
       -e 's|/work/session/dep8/\*.log|/work/logs/*.log|g' \
       -e 's|/work/bench_dep8_json|/work/bench_json|g' "$D/bench_sweep_v031.sh"
chmod +x "$D/bench_sweep_v031.sh"

for MC in 128 256; do
  say "  sweep MC=$MC"
  if [ "$MC" = 128 ]; then NPL='1 8 16 32 64 128'; else NPL='1 8 16 32 64 128 256 512'; fi
  MODEL=glm-5-3-flash-fp8 MC=$MC NP_LONG="$NPL" NP_SHORT='1 8 16 32 64 128 256 512 1024' \
    RESULTS_DIR="$D/results/mc$MC" bash "$D/bench_sweep_v031.sh" \
    || die "sweep MC=$MC failed"
done

say "PIPELINE COMPLETE"
echo "  functional : $D/results/functional.log"
echo "  needle     : $D/results/needle.log"
echo "  sweeps     : $D/results/mc128  $D/results/mc256"
echo "  server log : $D/logs/server.log"
echo
echo "Next (separate, needs the gate repo env):"
echo "  accuracy suite against http://127.0.0.1:8000/v1 with EVALSCOPE_PYTHON set"
sleep infinity
