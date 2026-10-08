#!/bin/bash
# Launch GLM-5.3-Flash DP8+EP8 on the v0.31.0-based image.
#
# Identical in every respect to the nightly-based recipe except the image, so
# any measured difference is attributable to the base version.
set -euo pipefail

IMAGE="${IMAGE:-glm53flash-vllm-v031:latest}"
NAME="${NAME:-glm53flash-v031}"
WORKDIR="${WORKDIR:-/home/psakhamo/GLM53_Flash_v031}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"
KV_DTYPE="${KV_DTYPE:-auto}"     # bf16. set fp8 only to exercise vllm#57134.

VIDEO_GID=$(getent group video  | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d --name "${NAME}" \
  --network host --device /dev/kfd --device /dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --ipc host --shm-size 128g \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK \
  --security-opt seccomp=unconfined \
  --ulimit nofile=1048576:1048576 \
  --entrypoint tail \
  -v /home/psakhamo/glm-5.3-flash-fp8:/mnt/models/GLM-5.3-Flash:ro \
  -v /home/psakhamo/aiter_jit_glm53_flash:/opt/vllm_cache \
  -v /home/psakhamo/aiter_home_glm53_flash:/root/.aiter \
  -v /home/psakhamo/fused_moe.py:/usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py:ro \
  -v "${WORKDIR}":/work \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_GCN_ARCH=gfx942 \
  -e VLLM_HANDSHAKE_TIMEOUT_MINS=30 \
  -e AITER_BATON_TIMEOUT=7200 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_SKINNY_GEMM=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e VLLM_ROCM_QR_INT4=1 \
  -e FLYDSL_FP8_MQA_LOGITS_VARIANT=mfma_r4_w4 \
  -e VLLM_KDA_FLYDSL=1 \
  -e VLLM_KDA_PREFILL_OUT=1 \
  -e VLLM_ROCM_SPARSE_MLA_GLUON=1 \
  -e VLLM_ROCM_SPARSE_MLA_GLUON_INVALID=0 \
  -e VLLM_GLM5NEXT_FUSED_KDA_DECODE=1 \
  "${IMAGE}" -f /dev/null

echo "container ${NAME} up from ${IMAGE}"

# Refuse to proceed on a silently-unpatched image.
docker exec "${NAME}" bash -lc '
P=/usr/local/lib/python3.12/dist-packages/vllm
grep -q "Storage-block specs" $P/v1/worker/utils.py \
  && echo "  PATCH OK #59412" || { echo "  MISSING #59412"; exit 1; }
grep -q "NoPE routing is geometry-based" $P/v1/attention/backends/mla/rocm_aiter_mla_sparse.py \
  && echo "  PATCH OK #57134 routing" || { echo "  MISSING #57134"; exit 1; }
grep -q "KV_IS_FP8" $P/v1/attention/ops/rocm_aiter_mla_sparse.py \
  && echo "  PATCH OK #57134 fp8 dequant" || { echo "  MISSING #57134"; exit 1; }
python3 -c "import vllm;print(\"  vllm\",vllm.__version__)"'

# The tuned kernels are env-gated AND depend on this patched AITER kernel.
docker exec "${NAME}" md5sum /usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py \
  | awk '{print "  fused_moe.py md5 "$1}'
md5sum /home/psakhamo/fused_moe.py | awk '{print "  host           md5 "$1" (must match)"}'

mkdir -p "${WORKDIR}/logs"
docker exec "${NAME}" bash -lc "
cd /work/logs
EXTRA=''
[ '${KV_DTYPE}' != 'auto' ] && EXTRA=\"--kv-cache-dtype ${KV_DTYPE}\"
setsid nohup vllm serve /mnt/models/GLM-5.3-Flash \
  --served-model-name glm-5-3-flash-fp8 \
  --data-parallel-size 8 --enable-expert-parallel \
  \$EXTRA \
  --trust-remote-code --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens 8192 --max-num-seqs 512 \
  --gpu-memory-utilization 0.8 --no-enable-prefix-caching \
  --distributed-timeout-seconds 7200 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --attention-backend ROCM_AITER_MLA_SPARSE \
  --port ${PORT} > /work/logs/server.log 2>&1 < /dev/null &
echo \"  serve pid=\$!\""

echo "  log: ${WORKDIR}/logs/server.log  (max-model-len ${MAX_MODEL_LEN}, kv ${KV_DTYPE})"
