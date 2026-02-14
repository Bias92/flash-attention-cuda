/*
 * flash_attention.cu
 *
 * Tiled FlashAttention forward kernel + naive baseline.
 * Target: SM with 128KB on-chip shared/L1 (RTX 4060 Ti / Jetson Orin).
 *
 * Tile config (Llama 3, head_dim=128):
 *   Br=64, Bc=64 => ~96.5KB shared < 128KB
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#define Br 64
#define Bc 64
#define D  128

/*
 * FlashAttention forward kernel.
 *
 * Each block processes Br rows of Q against all K/V tiles.
 * Grid: (ceil(N/Br), batch_size * num_heads)
 *
 * Shared memory layout:
 *   sQ:      Br x D   (FP16)  = 16 KB
 *   sK:      Bc x D   (FP16)  = 16 KB
 *   sV:      Bc x D   (FP16)  = 16 KB
 *   sS:      Br x Bc  (FP32)  = 16 KB   <- scores, never written to global mem
 *   sO:      Br x D   (FP32)  = 32 KB
 *   m, l:    Br each  (FP32)  = 0.5 KB
 *   Total: ~96.5 KB
 */
__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N,
    const float scale
) {
    int bh = blockIdx.y;
    int tile_q = blockIdx.x;
    int tid = threadIdx.x;

    int q_start = tile_q * Br;
    if (q_start >= N) return;
    int q_len = min(q_start + Br, N) - q_start;

    const half* Qh = Q + (size_t)bh * N * D;
    const half* Kh = K + (size_t)bh * N * D;
    const half* Vh = V + (size_t)bh * N * D;
    half*       Oh = O + (size_t)bh * N * D;

    extern __shared__ char smem[];

    half*  sQ = (half*)smem;
    half*  sK = (half*)(smem + Br * D * sizeof(half));
    half*  sV = (half*)(smem + (Br + Bc) * D * sizeof(half));
    float* sS = (float*)(smem + (Br + 2 * Bc) * D * sizeof(half));
    float* sO = (float*)((char*)sS + Br * Bc * sizeof(float));
    float* row_m = (float*)((char*)sO + Br * D * sizeof(float));
    float* row_l = row_m + Br;

    // load Q tile (once per block)
    for (int i = tid; i < Br * D; i += blockDim.x) {
        int r = i / D, c = i % D;
        sQ[i] = (r < q_len) ? Qh[(q_start + r) * D + c] : __float2half(0.0f);
    }

    for (int i = tid; i < Br * D; i += blockDim.x)
        sO[i] = 0.0f;

    if (tid < Br) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // iterate over K/V tiles
    int n_kv_tiles = (N + Bc - 1) / Bc;

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv_start = t * Bc;
        int kv_len = min(kv_start + Bc, N) - kv_start;

        // load K tile
        for (int i = tid; i < Bc * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            sK[i] = (r < kv_len) ? Kh[(kv_start + r) * D + c] : __float2half(0.0f);
        }

        // load V tile
        for (int i = tid; i < Bc * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            sV[i] = (r < kv_len) ? Vh[(kv_start + r) * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // compute S = Q @ K^T * scale
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int r = i / Bc, c = i % Bc;
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += __half2float(sQ[r * D + d]) * __half2float(sK[c * D + d]);

            sS[r * Bc + c] = (c < kv_len) ? dot * scale : -INFINITY;
        }
        __syncthreads();

        // online softmax + accumulate O
        if (tid < Br && tid < q_len) {
            float m_old = row_m[tid];
            float l_old = row_l[tid];

            // find new row max
            float m_new = m_old;
            for (int j = 0; j < Bc; j++) {
                float v = sS[tid * Bc + j];
                if (v > m_new) m_new = v;
            }

            // rescale previous accumulator
            float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);

            for (int d = 0; d < D; d++)
                sO[tid * D + d] *= alpha;

            float l_new = l_old * alpha;

            // accumulate exp(s - m_new) * V
            for (int j = 0; j < Bc; j++) {
                float p = (sS[tid * Bc + j] == -INFINITY)
                    ? 0.0f : expf(sS[tid * Bc + j] - m_new);

                l_new += p;
                for (int d = 0; d < D; d++)
                    sO[tid * D + d] += p * __half2float(sV[j * D + d]);
            }

            row_m[tid] = m_new;
            row_l[tid] = l_new;
        }
        __syncthreads();
    }

    // normalize and write back
    for (int i = tid; i < Br * D; i += blockDim.x) {
        int r = i / D, c = i % D;
        if (r < q_len) {
            float v = sO[r * D + c];
            float denom = row_l[r];
            Oh[(q_start + r) * D + c] = __float2half(denom > 0.0f ? v / denom : 0.0f);
        }
    }
}


void flash_attention_forward(
    const half* Q, const half* K, const half* V, half* O,
    int batch_heads, int N, int d, float scale
) {
    dim3 grid((N + Br - 1) / Br, batch_heads);
    dim3 block(128);

    size_t smem = Br * D * sizeof(half)      // sQ
               + Bc * D * sizeof(half)        // sK
               + Bc * D * sizeof(half)        // sV
               + Br * Bc * sizeof(float)      // sS
               + Br * D * sizeof(float)       // sO
               + Br * sizeof(float) * 2;      // m, l

    cudaFuncSetAttribute(
        flash_attn_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    flash_attn_fwd_kernel<<<grid, block, smem>>>(Q, K, V, O, N, scale);
}


/*
 * Naive attention baseline.
 * Materializes full N x N score matrix in global memory.
 */
__global__ void naive_attn_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ S_buf,
    const int N,
    const float scale
) {
    int bh = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* Qh = Q + (size_t)bh * N * D;
    const half* Kh = K + (size_t)bh * N * D;
    const half* Vh = V + (size_t)bh * N * D;
    half*       Oh = O + (size_t)bh * N * D;
    float*      Sh = S_buf + (size_t)bh * N * N;

    // S = Q @ K^T
    float mx = -INFINITY;
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += __half2float(Qh[row * D + d]) * __half2float(Kh[j * D + d]);
        dot *= scale;
        Sh[row * N + j] = dot;
        if (dot > mx) mx = dot;
    }

    // softmax
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float v = expf(Sh[row * N + j] - mx);
        Sh[row * N + j] = v;
        sum += v;
    }
    for (int j = 0; j < N; j++)
        Sh[row * N + j] /= sum;

    // O = S @ V
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += Sh[row * N + j] * __half2float(Vh[j * D + d]);
        Oh[row * D + d] = __float2half(acc);
    }
}

void naive_attention_forward(
    const half* Q, const half* K, const half* V, half* O,
    float* S_buf, int batch_heads, int N, int d, float scale
) {
    dim3 grid((N + 255) / 256, batch_heads);
    dim3 block(256);
    naive_attn_kernel<<<grid, block>>>(Q, K, V, O, S_buf, N, scale);
}
