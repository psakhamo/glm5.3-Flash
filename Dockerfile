# =============================================================================
# GLM-5.3-Flash on a CI-RELEASED vLLM base
#
# Base: vllm/vllm-openai-rocm:v0.31.0  (vllm 0.31.0+rocm723, released 2026-10-05)
#
# Replaces the previous recipe's nightly base
# (vllm/vllm-openai-rocm:nightly-ac68c3087..., vllm 0.30.1rc1.dev396+gac68c3087)
# so the stack is anchored to a tagged release rather than a moving commit.
#
# Base comparison, measured from both images:
#   python        3.12.14            (nightly: 3.12)
#   torch         2.13.0+git733fca1  (identical to nightly)
#   amd-aiter     0.1.23
#   vllm          0.31.0+rocm723
#   ROCM_AITER_MLA_SPARSE registered: yes
#   dist-packages /usr/local/lib/python3.12/dist-packages  (same as nightly)
#
# PATCHES. v0.31.0 was cut 2026-10-05; both PRs below were still open on
# 2026-10-06, and both were verified ABSENT from this image. Both apply cleanly
# (offsets only, no conflicts).
#
#   vllm#59412  [Bugfix][ROCm] Page-aligned kernel blocks for pooled indexers
#               REQUIRED. Without it long-context retrieval is broken: a
#               needle sweep on the previous base scored 1/12 unpatched vs
#               12/12 patched at 4K/16K/64K/118K, and the failure mode is a
#               WRONG value rather than a refusal.
#               file: vllm/v1/worker/utils.py
#
#   vllm#57134  [ROCm][Bugfix] FP8 KV cache for NoPE sparse MLA
#               Fixes a GPU memory-access fault under --kv-cache-dtype fp8 on
#               rope-free MLA. Inert as served here (bf16 KV), kept so the
#               option does not crash the GPU if someone sets it.
#               files: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
#                      vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
#
# Re-check both when a newer release lands: if either merges upstream, drop it
# from here rather than carrying a patch that is already in the base.
#
# Build:
#   docker build -f Dockerfile -t glm53flash-vllm-v031:latest .
# =============================================================================
FROM vllm/vllm-openai-rocm:v0.31.0

ARG VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

RUN cp ${VLLM_PKG}/v1/worker/utils.py ${VLLM_PKG}/v1/worker/utils.py.orig && \
    cp ${VLLM_PKG}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py \
       ${VLLM_PKG}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py.orig && \
    cp ${VLLM_PKG}/v1/attention/ops/rocm_aiter_mla_sparse.py \
       ${VLLM_PKG}/v1/attention/ops/rocm_aiter_mla_sparse.py.orig

COPY build/vllm/v1/worker/utils.py ${VLLM_PKG}/v1/worker/utils.py
COPY build/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py \
     ${VLLM_PKG}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
COPY build/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py \
     ${VLLM_PKG}/v1/attention/ops/rocm_aiter_mla_sparse.py

COPY patch/ /opt/patches/

# Fail the build unless BOTH patches are present and every file still compiles.
# A silently-unpatched image is worse than a failed build: #59412's absence
# produces confident wrong answers on long context rather than an error.
RUN python3 -m py_compile ${VLLM_PKG}/v1/worker/utils.py && \
    python3 -m py_compile ${VLLM_PKG}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py && \
    python3 -m py_compile ${VLLM_PKG}/v1/attention/ops/rocm_aiter_mla_sparse.py && \
    grep -q "Storage-block specs" ${VLLM_PKG}/v1/worker/utils.py && \
    grep -q "_block_size_is_supported" ${VLLM_PKG}/v1/worker/utils.py && \
    grep -q "NoPE routing is geometry-based" \
        ${VLLM_PKG}/v1/attention/backends/mla/rocm_aiter_mla_sparse.py && \
    grep -q "KV_IS_FP8" ${VLLM_PKG}/v1/attention/ops/rocm_aiter_mla_sparse.py && \
    find ${VLLM_PKG}/v1 -name '__pycache__' -type d -exec rm -rf {} + || true

LABEL vllm.base.image="vllm/vllm-openai-rocm:v0.31.0"
LABEL vllm.base.release="v0.31.0 (CI release, 2026-10-05)"
LABEL vllm.patch.pr1="https://github.com/vllm-project/vllm/pull/59412"
LABEL vllm.patch.pr2="https://github.com/vllm-project/vllm/pull/57134"
LABEL model="GLM-5.3-Flash FP8 (glm5_next), DP8+EP8"
