"""CUDA sources: 1-bit weights, residual INT8x3 activations, INT8 tensor-core MMA.

The kernel source strings are kept byte-identical to those of the research runtime the published numbers were
measured with. Comments inside them use its names: M1 is Kea-T (its ternary runtime is kea_kernels.ternary), MB1 is
Kea-B 0.0, R-hybrid is the hybrid stack (binary/hybrid.py).

Arithmetic (identical to the ternary expanded path, ternary.fused_kernels `_quantize` + `_int8_dot`):
  x is split into three INT8 residual parts with per-row scales s_c (scale = max(|r|max / 127,
  1e-30) with div.full, q = rni(r / scale) with div.full, r -= q * scale with mul.rn/sub.rn);
  acc_c[m, n] = sum_k q_c[m, k] * w[n, k] exactly in INT32 (w = +-1);
  y = fma(acc_2, s_2, fma(acc_0, s_0, acc_1 * s_1)) * alpha_n (+ bias_n) as mul.rn, fma.rn, fma.rn,
  mul.rn (or fma.rn with a bias), then optionally SiLU as y / (ex2.approx(-y * log2 e) + 1) with
  div.full, exactly the instruction sequence Triton emits for `_int8_dot` with SPLIT == 1. LLVM picks
  the first contraction per compile (BK = 64 compiles give fma(acc_1, s_1, acc_0 * s_0)); `order`
  selects either form.
  Every K split here is an INT32 reduction (exact), so results equal the SPLIT == 1 form bitwise.

Weight layout ("fragment layout", built once by `relayout`): uint32 [ceil(N/8), ceil(K/128), 32].
For n8 tile t and k128 block b, lane l = 4 * g + s of a warp holds one word whose bit 8 * i + p,
p = 2 * j + r, is the code of w[n = 8 t + g, k = 128 b + 32 j + 16 r + 4 s + i] (1 = +1). That is
exactly the B fragment of mma.m16n8k32 (.col): for k32 step j, register r of lane l is byte-plane
p of its word, expanded on chip to 0x01 / 0xFF per byte by ((w >> p) & 0x01010101) * -254 - 1.
A warp reads 128 contiguous bytes per (tile, k128 block); no INT8 weight exists in global memory.

b1_dot (DotConfig): CTA = WK groups of WM x WN warps, warp tile RM m16 x RN n8 for all three residual
parts; A (q) and the bit words share one cp.async pipeline of STAGES k128 x WK blocks (A swizzled for
ldmatrix.x4). K splits: WK warp groups inside the CTA (INT32 shared-memory reduction) and CLUSTER
CTAs along K with exchange XCHG: 0 INT32 over DSMEM (slow on SM120), 1 INT32 through an L2 workspace
behind barrier.cluster, 3 the three INT32 parts packed in one int64 (exact while 127 K < 2^20),
2 FP32 partials (not exact; unused), 4 deferred FP32 slices [CLUSTER, M, N]
for the hybrid row epilogue (no cluster; the base of kernels_exact's XCHG 5). PF = 1 loads the fragment epilogue's
scales before the main loop. Every configuration except XCHG 2 / 4 is bitwise the SPLIT == 1 result.
"""
from __future__ import annotations

import dataclasses

import torch

COMMON = r'''
typedef unsigned int u32;
typedef signed char i8;
typedef unsigned char u8;
#define DEVI __device__ __forceinline__

DEVI u32 smem_u32(const void *p) {
    u32 a; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a;
}
DEVI void cp_async16(u32 dst, const void *src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
DEVI u32 lds32(u32 addr) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory"); return v; }
DEVI void sts64(u32 addr, int a, int b) {
    asm volatile("st.shared.v2.s32 [%0], {%1, %2};" :: "r"(addr), "r"(a), "r"(b) : "memory");
}
DEVI void ldmatrix_x4(u32 &r0, u32 &r1, u32 &r2, u32 &r3, u32 addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
DEVI void mma_s8(int *c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Byte plane P of a fragment word -> four INT8 weights: bit 1 -> 0x01 (+1), bit 0 -> 0xFF (-1).
template <int P> DEVI u32 expand(u32 w) { return 0xFFFFFFFFu - ((w >> P) & 0x01010101u) * 254u; }

DEVI float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float ex2_approx(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }

// M1's SPLIT == 1 epilogue, op for op (see module docstring). flags bit 0: SiLU. LLVM contracts
// Triton's acc0 * s0 + acc1 * s1 either way depending on the compile: flags bit 1 clear gives
// fma(acc0, s0, acc1 * s1) (M1's BK = 128 kernels), set gives fma(acc1, s1, acc0 * s0) (BK = 64).
DEVI float epilogue_v(int a0, int a1, int a2, float s0, float s1, float s2, float alpha, bool has_bias, float bias, int flags) {
    const int silu = flags & 1;
    float r = (flags & 2) ? __fmaf_rn(__int2float_rn(a1), s1, __fmul_rn(__int2float_rn(a0), s0))
                          : __fmaf_rn(__int2float_rn(a0), s0, __fmul_rn(__int2float_rn(a1), s1));
    r = __fmaf_rn(__int2float_rn(a2), s2, r);
    r = has_bias ? __fmaf_rn(r, alpha, bias) : __fmul_rn(r, alpha);
    if (silu) r = div_full(r, __fadd_rn(ex2_approx(__fmul_rn(-r, 1.44269502162933349609375f)), 1.0f));
    return r;
}
DEVI float epilogue(int a0, int a1, int a2, float s0, float s1, float s2, float alpha, const float *bias, int n, int flags) {
    const int silu = flags & 1;
    float r = (flags & 2) ? __fmaf_rn(__int2float_rn(a1), s1, __fmul_rn(__int2float_rn(a0), s0))
                          : __fmaf_rn(__int2float_rn(a0), s0, __fmul_rn(__int2float_rn(a1), s1));
    r = __fmaf_rn(__int2float_rn(a2), s2, r);
    r = bias ? __fmaf_rn(r, alpha, bias[n]) : __fmul_rn(r, alpha);
    if (silu) r = div_full(r, __fadd_rn(ex2_approx(__fmul_rn(-r, 1.44269502162933349609375f)), 1.0f));
    return r;
}

// M1's split-K partial (fused_kernels._int8_dot with SPLIT > 1, order 0) and _finish:
// P = fma(acc2, s2, fma(acc0, s0, acc1 * s1)); y = (((0 + P_0) + P_1) + ...) * alpha (+ bias), then SiLU.
DEVI float partial(int a0, int a1, int a2, float s0, float s1, float s2) {
    return __fmaf_rn(__int2float_rn(a2), s2, __fmaf_rn(__int2float_rn(a0), s0, __fmul_rn(__int2float_rn(a1), s1)));
}
DEVI float finish(float sum, float alpha, const float *bias, int n, int flags) {
    float r = bias ? __fmaf_rn(sum, alpha, bias[n]) : __fmul_rn(sum, alpha);
    if (flags & 1) r = div_full(r, __fadd_rn(ex2_approx(__fmul_rn(-r, 1.44269502162933349609375f)), 1.0f));
    return r;
}
// Three exact INT32 parts in one int64 (21-bit signed lanes): linear, so sums of packed values are the
// packed sums; exact while every lane total stays in [-2^20, 2^20), i.e. 127 * K < 2^20 (K <= 8256).
DEVI long long pack3(int a0, int a1, int a2) { return (long long)a0 + (long long)a1 * (1LL << 21) + (long long)a2 * (1LL << 42); }
DEVI void unpack3(long long p, int &a0, int &a1, int &a2) {
    a0 = (int)((p << 43) >> 43);
    const long long r = (p - a0) >> 21;
    a1 = (int)((r << 43) >> 43);
    a2 = (int)((r - a1) >> 21);
}
DEVI void ld_cg_v2ll(const long long *p, long long &a, long long &b) {
    asm volatile("ld.global.cg.v2.s64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p));
}
DEVI void ld_cg_v4f(const float *p, float &a, float &b, float &c, float &d) {
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(p));
}

DEVI u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVI u32 map_rank(u32 addr, u32 rank) { u32 r; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r; }
DEVI void ld_cg_v4(const int *p, int &a, int &b, int &c, int &d) {  // L2 only: peers wrote it
    asm volatile("ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p));
}
DEVI void ld_cluster_v4(u32 addr, int &a, int &b, int &c, int &d) {  // ordered after cluster_sync (volatile)
    asm volatile("ld.shared::cluster.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
'''

DOT = r'''
// One CTA: BMR rows (all three residual parts) x BN columns x the k128 blocks of its cluster rank.
// Warps: WK groups of WM x WN; warp tile (RM m16) x (RN n8) for every residual part. Group wk takes
// k128 block wk of every WK-block pipeline stage; groups are reduced in INT32 through shared memory.
template <int WM, int WN, int WK, int RM, int RN, int STAGES, int CLUSTER, int FLAGS, int XCHG, int PF>
DEVI void b1_dot(const i8 *__restrict__ Q, long long cstride, int ldq, int kq,
                 const float *__restrict__ QS, int qs_stride,
                 const u32 *__restrict__ Wb, int nt, int kb_total,
                 const float *__restrict__ WS, const float *__restrict__ Bias,
                 float *__restrict__ Y, int ldy, int M, int N, int *__restrict__ WSP, unsigned char *smem) {
    constexpr int THREADS = WM * WN * WK * 32;
    constexpr int BMR = WM * RM * 16;
    constexpr int BN = WN * RN * 8;
    constexpr int NT = BN / 8;
    constexpr int A_SUB = 3 * BMR * 128;          // one k128 block of A (all three parts)
    constexpr int A_BYTES = A_SUB * WK;
    constexpr int STAGE_BYTES = A_BYTES + NT * 128 * WK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wk = warp / (WM * WN), wm = (warp % (WM * WN)) / WN, wn = warp % WN;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BMR;
    // XCHG 4 (deferred) splits K over blockIdx.z without a cluster; otherwise the cluster rank does.
    const int rank = CLUSTER > 1 ? (XCHG == 4 ? (int)blockIdx.z : (int)cluster_rank()) : 0;
    const int kb0 = (int)(((long long)kb_total * rank) / CLUSTER);
    const int kb1 = (int)(((long long)kb_total * (rank + 1)) / CLUSTER);
    const int nblocks = (kb1 - kb0 + WK - 1) / WK;  // pipeline stages of WK k128 blocks
    const u32 sbase = smem_u32(smem);
    // PF: the fragment epilogue's scales (CLUSTER == 1, XCHG 2 / 4) are loaded here, so their L2 latency
    // overlaps the pipeline prologue instead of trailing the main loop (tuned per config: +registers).
    constexpr bool FRAG = PF && (CLUSTER == 1 || XCHG == 2 || XCHG == 4);
    float pq[FRAG ? RM : 1][2][3], pw[FRAG ? RN : 1][2], pb[FRAG ? RN : 1][2];
    if constexpr (FRAG) {
        const int g0 = lane >> 2, t0 = lane & 3;
        #pragma unroll
        for (int i = 0; i < RM; ++i)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + (wm * RM + i) * 16 + g0 + 8 * h;
                const bool ok = m < M;
                pq[i][h][0] = ok ? QS[m] : 0.f;
                pq[i][h][1] = ok ? QS[qs_stride + m] : 0.f;
                pq[i][h][2] = ok ? QS[2 * qs_stride + m] : 0.f;
            }
        #pragma unroll
        for (int j = 0; j < RN; ++j)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int n = n0 + (wn * RN + j) * 8 + t0 * 2 + e;
                pw[j][e] = n < N ? WS[n] : 0.f;
                pb[j][e] = (Bias && n < N) ? Bias[n] : 0.f;
            }
    }

    auto issue = [&](int blk, int stage) {
        const u32 st = sbase + stage * STAGE_BYTES;
        #pragma unroll 4
        for (int i = tid; i < 3 * BMR * 8 * WK; i += THREADS) {
            const int w = i / (3 * BMR * 8), rest = i - w * (3 * BMR * 8);
            const int row = rest >> 3, ch = rest & 7;
            const int c = row / BMR, r = row - c * BMR;
            const int kb = kb0 + blk * WK + w;
            const int m = m0 + r, k = kb * 128 + ch * 16;
            const bool ok = m < M && k < kq && kb < kb1;
            const i8 *src = ok ? Q + c * cstride + (long long)m * ldq + k : Q;
            cp_async16(st + w * A_SUB + row * 128 + ((ch ^ (row & 7)) << 4), src, ok ? 16 : 0);
        }
        for (int i = tid; i < NT * 8 * WK; i += THREADS) {
            const int w = i / (NT * 8), rest = i - w * (NT * 8);
            const int t = rest >> 3, ch = rest & 7, tile = (n0 >> 3) + t;
            const int kb = kb0 + blk * WK + w;
            const bool ok = tile < nt && kb < kb1;
            const u32 *src = ok ? Wb + ((long long)tile * kb_total + kb) * 32 + ch * 4 : Wb;
            cp_async16(st + A_BYTES + (w * NT + t) * 128 + ch * 16, src, ok ? 16 : 0);
        }
    };

    int acc[3][RM][RN][4];
    #pragma unroll
    for (int c = 0; c < 3; ++c)
        #pragma unroll
        for (int i = 0; i < RM; ++i)
            #pragma unroll
            for (int j = 0; j < RN; ++j)
                #pragma unroll
                for (int e = 0; e < 4; ++e) acc[c][i][j][e] = 0;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nblocks) issue(s, s);
        cp_commit();
    }
    for (int blk = 0; blk < nblocks; ++blk) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (blk + STAGES - 1 < nblocks) issue(blk + STAGES - 1, (blk + STAGES - 1) % STAGES);
        cp_commit();
        const u32 st = sbase + (blk % STAGES) * STAGE_BYTES + wk * A_SUB;  // this group's A block
        const u32 sb = sbase + (blk % STAGES) * STAGE_BYTES + A_BYTES + wk * NT * 128;
        u32 bw[RN];
        #pragma unroll
        for (int j = 0; j < RN; ++j) bw[j] = lds32(sb + (wn * RN + j) * 128 + lane * 4);
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            u32 b0[RN], b1[RN];
            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                switch (ks) {
                    case 0: b0[j] = expand<0>(bw[j]); b1[j] = expand<1>(bw[j]); break;
                    case 1: b0[j] = expand<2>(bw[j]); b1[j] = expand<3>(bw[j]); break;
                    case 2: b0[j] = expand<4>(bw[j]); b1[j] = expand<5>(bw[j]); break;
                    default: b0[j] = expand<6>(bw[j]); b1[j] = expand<7>(bw[j]); break;
                }
            }
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                #pragma unroll
                for (int i = 0; i < RM; ++i) {
                    const int row = c * BMR + (wm * RM + i) * 16 + (lane & 15);
                    const int ch = ks * 2 + (lane >> 4);
                    u32 a0, a1, a2, a3;
                    ldmatrix_x4(a0, a1, a2, a3, st + row * 128 + ((ch ^ (row & 7)) << 4));
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) mma_s8(acc[c][i][j], a0, a1, a2, a3, b0[j], b1[j]);
                }
            }
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, s4 = lane & 3;
    if constexpr (WK > 1) {  // exact INT32 reduction of the K groups into group 0's registers
        constexpr int PST = BN + 8;
        constexpr int SLICE = 3 * BMR * PST;
        __syncthreads();
        if (wk > 0) {
            #pragma unroll
            for (int c = 0; c < 3; ++c)
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j)
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int row = c * BMR + (wm * RM + i) * 16 + g + 8 * h, col = (wn * RN + j) * 8 + s4 * 2;
                            sts64(sbase + ((wk - 1) * SLICE + row * PST + col) * 4, acc[c][i][j][2 * h], acc[c][i][j][2 * h + 1]);
                        }
        }
        __syncthreads();
        if (wk == 0) {
            const int *red = reinterpret_cast<const int *>(smem);
            #pragma unroll
            for (int w = 0; w < WK - 1; ++w)
                #pragma unroll
                for (int c = 0; c < 3; ++c)
                    #pragma unroll
                    for (int i = 0; i < RM; ++i)
                        #pragma unroll
                        for (int j = 0; j < RN; ++j)
                            #pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                const int row = c * BMR + (wm * RM + i) * 16 + g + 8 * h, col = (wn * RN + j) * 8 + s4 * 2;
                                const int2 v = *reinterpret_cast<const int2 *>(red + w * SLICE + row * PST + col);
                                acc[c][i][j][2 * h] += v.x;
                                acc[c][i][j][2 * h + 1] += v.y;
                            }
        }
        __syncthreads();  // the shared memory is reused by the exchange below
    }
    if constexpr (CLUSTER == 1) {
        if (wk != 0) return;
        const bool pairs = (N % 2 == 0) && (ldy % 2 == 0);
        #pragma unroll
        for (int i = 0; i < RM; ++i) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + (wm * RM + i) * 16 + g + 8 * h;
                if (m >= M) continue;
                float q0, q1, q2;
                if constexpr (FRAG) { q0 = pq[i][h][0]; q1 = pq[i][h][1]; q2 = pq[i][h][2]; }
                else { q0 = QS[m]; q1 = QS[qs_stride + m]; q2 = QS[2 * qs_stride + m]; }
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    const int n = n0 + (wn * RN + j) * 8 + s4 * 2;
                    float *out = Y + (long long)m * ldy + n;
                    if (pairs && n + 1 < N) {
                        float2 v;
                        if constexpr (FRAG) {
                            v.x = epilogue_v(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h], q0, q1, q2, pw[j][0], Bias != nullptr, pb[j][0], FLAGS);
                            v.y = epilogue_v(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1], q0, q1, q2, pw[j][1], Bias != nullptr, pb[j][1], FLAGS);
                        } else {
                            v.x = epilogue(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h], q0, q1, q2, WS[n], Bias, n, FLAGS);
                            v.y = epilogue(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1], q0, q1, q2, WS[n + 1], Bias, n + 1, FLAGS);
                        }
                        *reinterpret_cast<float2 *>(out) = v;
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 2; ++e)
                            if (n + e < N) {
                                if constexpr (FRAG)
                                    out[e] = epilogue_v(acc[0][i][j][2 * h + e], acc[1][i][j][2 * h + e], acc[2][i][j][2 * h + e], q0, q1, q2, pw[j][e], Bias != nullptr, pb[j][e], FLAGS);
                                else
                                    out[e] = epilogue(acc[0][i][j][2 * h + e], acc[1][i][j][2 * h + e], acc[2][i][j][2 * h + e], q0, q1, q2, WS[n + e], Bias, n + e, FLAGS);
                            }
                    }
                }
            }
        }
    } else if constexpr (XCHG == 4) {
        // Deferred split-K (R-hybrid's protocol): slice z writes M1's FP32 split partial
        // P = fma(acc2, s2, fma(acc0, s0, acc1 * s1)) to Y + z * M * ldy, i.e. [CLUSTER, M, N];
        // inference.row_fusion._row_epilogue sums the slices, applies alpha / bias, the residual,
        // the next LayerNorm and the next INT8x3 quantization in one kernel (as for ptx_row_best).
        if (wk == 0) {
            float *slice = Y + (long long)rank * M * ldy;
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m = m0 + (wm * RM + i) * 16 + g + 8 * h;
                    if (m >= M) continue;
                    float q0, q1, q2;
                    if constexpr (FRAG) { q0 = pq[i][h][0]; q1 = pq[i][h][1]; q2 = pq[i][h][2]; }
                    else { q0 = QS[m]; q1 = QS[qs_stride + m]; q2 = QS[2 * qs_stride + m]; }
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        const int n = n0 + (wn * RN + j) * 8 + s4 * 2;
                        float *out = slice + (long long)m * ldy + n;
                        const float a = partial(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h], q0, q1, q2);
                        const float b = partial(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1], q0, q1, q2);
                        if (n + 1 < N && (ldy % 2 == 0)) {
                            *reinterpret_cast<float2 *>(out) = make_float2(a, b);
                        } else {
                            if (n < N) out[0] = a;
                            if (n + 1 < N) out[1] = b;
                        }
                    }
                }
        }
    } else if constexpr (XCHG == 3) {
        // Exact INT32 split-K with the three parts packed into one int64 per element (2/3 of XCHG 1's
        // L2 volume): rank partials to the workspace, cluster barrier, rank r sums its share in int64.
        constexpr int TILE = BMR * BN;
        long long *base = reinterpret_cast<long long *>(WSP) + ((long long)(blockIdx.y * gridDim.x + blockIdx.x) * CLUSTER) * TILE;
        if (wk == 0) {
            long long *mine = base + (long long)rank * TILE;
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = (wm * RM + i) * 16 + g + 8 * h;
                    if (m0 + row >= M) continue;
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        const int col = (wn * RN + j) * 8 + s4 * 2;
                        longlong2 v;
                        v.x = pack3(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h]);
                        v.y = pack3(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1]);
                        *reinterpret_cast<longlong2 *>(mine + row * BN + col) = v;
                    }
                }
        }
        cluster_sync();
        constexpr int GROUPS = BMR * BN / 4;
        const int first = (GROUPS * rank) / CLUSTER, last = (GROUPS * (rank + 1)) / CLUSTER;
        const bool quads = (N % 4 == 0) && (ldy % 4 == 0);
        for (int gi = first + tid; gi < last; gi += THREADS) {
            const int row = gi / (BN / 4), col = (gi % (BN / 4)) * 4;
            const int m = m0 + row, n = n0 + col;
            if (m >= M || n >= N) continue;
            long long part[CLUSTER][4];
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r) {
                const long long *src = base + (long long)((rank + r) % CLUSTER) * TILE + row * BN + col;
                ld_cg_v2ll(src, part[r][0], part[r][1]);
                ld_cg_v2ll(src + 2, part[r][2], part[r][3]);
            }
            const float q0 = QS[m], q1 = QS[qs_stride + m], q2 = QS[2 * qs_stride + m];
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                long long sum = 0;
                #pragma unroll
                for (int r = 0; r < CLUSTER; ++r) sum += part[r][e];
                int a0, a1, a2;
                unpack3(sum, a0, a1, a2);
                v[e] = n + e < N ? epilogue(a0, a1, a2, q0, q1, q2, WS[n + e], Bias, n + e, FLAGS) : 0.f;
            }
            float *out = Y + (long long)m * ldy + n;
            if (quads) {
                *reinterpret_cast<float4 *>(out) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (n + e < N) out[e] = v[e];
            }
        }
    } else if constexpr (XCHG == 2) {
        // FP32 split-K over the cluster (M1's split arithmetic, deterministic, 1/3 of the INT32 volume):
        // each rank writes P_rank for its valid rows to an L2 workspace, cluster barrier, then rank r
        // finishes a 1/CLUSTER share: (((0 + P_0) + P_1) + ...) * alpha (+ bias), in rank order.
        constexpr int TILE = BMR * BN;
        float *wsp = reinterpret_cast<float *>(WSP) + ((long long)(blockIdx.y * gridDim.x + blockIdx.x) * CLUSTER + rank) * TILE;
        if (wk == 0) {
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = (wm * RM + i) * 16 + g + 8 * h, m = m0 + row;
                    if (m >= M) continue;
                    const float q0 = QS[m], q1 = QS[qs_stride + m], q2 = QS[2 * qs_stride + m];
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        const int col = (wn * RN + j) * 8 + s4 * 2;
                        *reinterpret_cast<float2 *>(wsp + row * BN + col) = make_float2(
                            partial(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h], q0, q1, q2),
                            partial(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1], q0, q1, q2));
                    }
                }
        }
        cluster_sync();
        const float *base = reinterpret_cast<const float *>(WSP) + ((long long)(blockIdx.y * gridDim.x + blockIdx.x) * CLUSTER) * TILE;
        constexpr int GROUPS = BMR * BN / 4;
        const int first = (GROUPS * rank) / CLUSTER, last = (GROUPS * (rank + 1)) / CLUSTER;
        const bool quads = (N % 4 == 0) && (ldy % 4 == 0);
        for (int gi = first + tid; gi < last; gi += THREADS) {
            const int row = gi / (BN / 4), col = (gi % (BN / 4)) * 4;
            const int m = m0 + row, n = n0 + col;
            if (m >= M || n >= N) continue;
            float part[CLUSTER][4];
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r)
                ld_cg_v4f(base + (long long)r * TILE + row * BN + col, part[r][0], part[r][1], part[r][2], part[r][3]);
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float sum = __fadd_rn(0.0f, part[0][e]);
                #pragma unroll
                for (int r = 1; r < CLUSTER; ++r) sum = __fadd_rn(sum, part[r][e]);
                v[e] = n + e < N ? finish(sum, WS[n + e], Bias, n + e, FLAGS) : 0.f;
            }
            float *out = Y + (long long)m * ldy + n;
            if (quads) {
                *reinterpret_cast<float4 *>(out) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (n + e < N) out[e] = v[e];
            }
        }
    } else {
        // Exact INT32 split-K over the cluster. XCHG 0: every rank parks its partial tile in its own
        // shared memory and peers read it over DSMEM; XCHG 1: ranks write partials to an L2
        // workspace (valid rows only) and peers read them back with ld.global.cg. Either way the
        // cluster barrier (release / acquire, cluster scope) orders the writes before the reads,
        // then rank r reduces a 1/CLUSTER share of the tile over all ranks and applies the epilogue.
        constexpr int PST = BN + 8;  // shared row stride in int32 (+8 spreads fragment stores over banks)
        constexpr int TILE = 3 * BMR * BN;
        int *wsp = XCHG == 1 ? WSP + ((long long)(blockIdx.y * gridDim.x + blockIdx.x) * CLUSTER) * TILE : nullptr;
        if constexpr (XCHG == 0) __syncthreads();
        if (wk == 0) {
            #pragma unroll
            for (int c = 0; c < 3; ++c)
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j)
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int row = (wm * RM + i) * 16 + g + 8 * h, col = (wn * RN + j) * 8 + s4 * 2;
                            if constexpr (XCHG == 0) {
                                sts64(sbase + ((c * BMR + row) * PST + col) * 4, acc[c][i][j][2 * h], acc[c][i][j][2 * h + 1]);
                            } else if (m0 + row < M) {
                                *reinterpret_cast<int2 *>(wsp + (long long)rank * TILE + (c * BMR + row) * BN + col) =
                                    make_int2(acc[c][i][j][2 * h], acc[c][i][j][2 * h + 1]);
                            }
                        }
        }
        cluster_sync();
        constexpr int GROUPS = BMR * BN / 4;
        const int first = (GROUPS * rank) / CLUSTER, last = (GROUPS * (rank + 1)) / CLUSTER;
        const bool quads = (N % 4 == 0) && (ldy % 4 == 0);
        for (int gi = first + tid; gi < last; gi += THREADS) {
            const int row = gi / (BN / 4), col = (gi % (BN / 4)) * 4;
            const int m = m0 + row, n = n0 + col;
            if (m >= M || n >= N) continue;
            // Issue every load before the first use so all CLUSTER x 3 reads are in flight.
            int part[CLUSTER][3][4];
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r) {
                const int rr = (rank + r) % CLUSTER;  // stagger peers
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    if constexpr (XCHG == 0) {
                        ld_cluster_v4(map_rank(sbase + ((c * BMR + row) * PST + col) * 4, rr),
                                      part[r][c][0], part[r][c][1], part[r][c][2], part[r][c][3]);
                    } else {
                        ld_cg_v4(wsp + (long long)rr * TILE + (c * BMR + row) * BN + col,
                                 part[r][c][0], part[r][c][1], part[r][c][2], part[r][c][3]);
                    }
                }
            }
            int sum[3][4] = {};
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r)
                #pragma unroll
                for (int c = 0; c < 3; ++c)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) sum[c][e] += part[r][c][e];
            const float q0 = QS[m], q1 = QS[qs_stride + m], q2 = QS[2 * qs_stride + m];
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                v[e] = n + e < N ? epilogue(sum[0][e], sum[1][e], sum[2][e], q0, q1, q2, WS[n + e], Bias, n + e, FLAGS) : 0.f;
            float *out = Y + (long long)m * ldy + n;
            if (quads) {
                *reinterpret_cast<float4 *>(out) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (n + e < N) out[e] = v[e];
            }
        }
        if constexpr (XCHG == 0) cluster_sync();  // no CTA may exit while a peer still reads its shared memory
    }
}
'''

DOT_ENTRY = r'''
extern "C" __global__ void __launch_bounds__({threads}) {cluster_attr}
{name}(const i8 *Q, long long cstride, int ldq, int kq, const float *QS, int qs_stride,
       const u32 *Wb, int nt, int kb_total, const float *WS, const float *Bias,
       float *Y, int ldy, int M, int N, int *WSP) {{
    extern __shared__ __align__(128) unsigned char smem[];
    b1_dot<{wm}, {wn}, {wk}, {rm}, {rn}, {stages}, {cluster}, {flags}, {xchg}, {pf}>(Q, cstride, ldq, kq, QS, qs_stride, Wb, nt,
                                                       kb_total, WS, Bias, Y, ldy, M, N, WSP, smem);
}}
'''

QUANTIZE = r'''
// Bitwise the arithmetic of M1's inference.fused_kernels._quantize without normalization/SiLU: one
// warp per row; x[m, k] at X + (m / T) * XB + (m % T) * XT + k * XK; writes q [3][M][ldq] (zero
// padded from K to ldq) and qs [3][M]. Max reductions are exact, so their order is free.
DEVI float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
DEVI float part_scale(float amax) { return fmaxf(div_full(amax, 127.0f), 1.0e-30f); }

extern "C" __global__ void __launch_bounds__(128) b1_quantize(
        const float *__restrict__ X, long long XB, long long XT, long long XK, int T,
        i8 *__restrict__ Qo, float *__restrict__ QSo, int M, int K, int ldq) {
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (m >= M) return;
    const float *x = X + (long long)(m / T) * XB + (long long)(m % T) * XT;
    float amax = 0.f;
    for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(x[k * XK]));
    const float s0 = part_scale(warp_max(amax));
    amax = 0.f;
    for (int k = lane; k < K; k += 32) {
        float v = x[k * XK];
        float q = rintf(div_full(v, s0));
        amax = fmaxf(amax, fabsf(__fsub_rn(v, __fmul_rn(q, s0))));
    }
    const float s1 = part_scale(warp_max(amax));
    amax = 0.f;
    for (int k = lane; k < K; k += 32) {
        float v = x[k * XK];
        float q = rintf(div_full(v, s0)); v = __fsub_rn(v, __fmul_rn(q, s0));
        q = rintf(div_full(v, s1)); v = __fsub_rn(v, __fmul_rn(q, s1));
        amax = fmaxf(amax, fabsf(v));
    }
    const float s2 = part_scale(warp_max(amax));
    const long long plane = (long long)M * ldq;
    i8 *q0 = Qo + (long long)m * ldq;
    for (int k = lane; k < ldq; k += 32) {
        int a = 0, b = 0, c = 0;
        if (k < K) {
            float v = x[k * XK];
            float q = rintf(div_full(v, s0)); a = (int)q; v = __fsub_rn(v, __fmul_rn(q, s0));
            q = rintf(div_full(v, s1)); b = (int)q; v = __fsub_rn(v, __fmul_rn(q, s1));
            q = rintf(div_full(v, s2)); c = (int)q;
        }
        q0[k] = (i8)a; q0[plane + k] = (i8)b; q0[2 * plane + k] = (i8)c;
    }
    if (lane == 0) { QSo[m] = s0; QSo[M + m] = s1; QSo[2 * M + m] = s2; }
}
'''

FALLBACK = r'''
// Any shape: weights in MB1's own layout (uint8 [N, row_bytes], LSB first). One thread per output.
extern "C" __global__ void __launch_bounds__(128) b1_fallback(
        const i8 *__restrict__ Q, long long cstride, int ldq, const float *__restrict__ QS, int qs_stride,
        const u8 *__restrict__ codes, int row_bytes, const float *__restrict__ WS, const float *__restrict__ Bias,
        float *__restrict__ Y, int ldy, int M, int N, int K, int silu) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * N) return;
    const int m = (int)(idx / N), n = (int)(idx % N);
    const i8 *q = Q + (long long)m * ldq;
    const u8 *w = codes + (long long)n * row_bytes;
    int a0 = 0, a1 = 0, a2 = 0;
    for (int k = 0; k < K; ++k) {
        const int sign = ((w[k >> 3] >> (k & 7)) & 1) ? 1 : -1;
        a0 += sign * q[k]; a1 += sign * q[cstride + k]; a2 += sign * q[2 * cstride + k];
    }
    Y[(long long)m * ldy + n] = epilogue(a0, a1, a2, QS[m], QS[qs_stride + m], QS[2 * qs_stride + m], WS[n], Bias, n, silu);
}
'''


@dataclasses.dataclass(frozen=True)
class DotConfig:
    """CTA = WM x WN warps; warp tile RM m16 x RN n8 per residual part; K split over a CLUSTER."""
    wm: int
    wn: int
    rm: int
    rn: int
    stages: int
    cluster: int
    xchg: int = 0  # cluster split-K exchange: 0 INT32 via DSMEM, 1 INT32 via L2, 2 FP32 partial via L2
    #               (not exact; unused), 3 INT32x3 packed in int64 via L2 (exact),
    #               4 deferred: FP32 split partials [cluster, M, N] for the hybrid row epilogue (no cluster)
    wk: int = 1    # warp groups splitting each stage's K inside the CTA (exact INT32 smem reduction)
    pf: int = 0    # 1: prefetch the fragment epilogue's scales before the main loop (CLUSTER 1, XCHG 2 / 4)

    @property
    def threads(self) -> int:
        return self.wm * self.wn * self.wk * 32

    @property
    def bmr(self) -> int:
        return self.wm * self.rm * 16

    @property
    def bn(self) -> int:
        return self.wn * self.rn * 8

    @property
    def shared_bytes(self) -> int:
        stage = (3 * self.bmr * 128 + (self.bn // 8) * 128) * self.wk
        reduce = 3 * self.bmr * (self.bn + 8) * 4 if self.cluster > 1 and self.xchg == 0 else 0
        groups = (self.wk - 1) * 3 * self.bmr * (self.bn + 8) * 4
        return max(self.stages * stage, reduce, groups)

    def grid(self, m: int, n: int) -> tuple[int, int, int]:
        return (-(-n // self.bn), -(-m // self.bmr), self.cluster)

    def label(self) -> str:
        return (f"w{self.wm}x{self.wn}" + (f"x{self.wk}" if self.wk > 1 else "") +
                f"_r{self.rm}x{self.rn}_s{self.stages}_c{self.cluster}" + ("", "_g", "_f", "_p", "_d")[self.xchg]
                + ("_pf" if self.pf else ""))

    @property
    def exact(self) -> bool:
        """Bitwise the SPLIT == 1 result (INT32-exact K reduction)."""
        return self.cluster == 1 or self.xchg not in (2, 4)

    def workspace_ints(self, m: int, n: int) -> int:
        gx, gy, _ = self.grid(m, n)
        if self.cluster == 1 or self.xchg == 0:
            return 0
        if self.xchg == 4:
            return 0
        return gx * gy * self.cluster * {1: 3, 2: 1, 3: 2}[self.xchg] * self.bmr * self.bn

    def validate(self) -> None:
        if not (self.wm >= 1 and self.wn >= 1 and self.rm >= 1 and self.rn >= 1 and self.stages >= 2
                and 1 <= self.cluster <= 8 and self.threads <= 1024 and self.xchg in (0, 1, 2, 3, 4) and self.wk >= 1 and self.pf in (0, 1)):
            raise ValueError(f"invalid config {self}")
        if self.shared_bytes > 101376:
            raise ValueError(f"config {self.label()} needs {self.shared_bytes} B of shared memory")


def dot_source(config: DotConfig, silu: bool, order: int = 0) -> tuple[str, str]:
    config.validate()
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    name = f"b1_dot_{config.label()}_{'silu' if silu else 'id'}_o{order}"
    entry = DOT_ENTRY.format(threads=config.threads, name=name, wm=config.wm, wn=config.wn, rm=config.rm,
                             rn=config.rn, stages=config.stages, cluster=config.cluster, flags=int(silu) | 2 * order, xchg=config.xchg,
                             wk=config.wk, pf=config.pf,
                             cluster_attr=(f"__cluster_dims__(1, 1, {config.cluster})"
                                           if config.cluster > 1 and config.xchg != 4 else ""))
    return COMMON + DOT + entry, name


def quantize_source() -> tuple[str, str]:
    return COMMON + QUANTIZE, "b1_quantize"


def fallback_source() -> tuple[str, str]:
    return COMMON + FALLBACK, "b1_fallback"


# --- weight layout --------------------------------------------------------------------------------

def fragment_bytes(n: int, k: int) -> int:
    return -(-n // 8) * -(-k // 128) * 128


def relayout(bits: torch.Tensor) -> torch.Tensor:
    """uint8 {0, 1} [N, K] (1 = +1) -> int32 [ceil(N/8), ceil(K/128), 32] fragment words (see docstring).

    Padding (rows >= N, k >= K) is 0; the kernels never use it (A is zero there, outputs masked)."""
    n, k = bits.shape
    nt, kb = -(-n // 8), -(-k // 128)
    padded = torch.zeros(nt * 8, kb * 128, dtype=torch.int64, device=bits.device)
    padded[:n, :k] = bits.to(torch.int64)
    # [t, g, b, j, r, s, i] -> [t, b, g, s, i, j, r]; bit index 8 i + 2 j + r
    v = padded.view(nt, 8, kb, 4, 2, 4, 4).permute(0, 2, 1, 5, 6, 3, 4).reshape(nt, kb, 32, 32)
    shifts = torch.arange(32, dtype=torch.int64, device=bits.device)
    words = (v << shifts).sum(dim=-1)  # disjoint bits: sum == OR, < 2**32
    return (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32).contiguous()


def unlayout(words: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of relayout (tests): int32 fragment words -> uint8 {0, 1} [N, K]."""
    nt, kb, _ = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, dtype=torch.int64, device=words.device)
    v = ((w[..., None] >> shifts) & 1).view(nt, kb, 8, 4, 4, 4, 2)  # [t, b, g, s, i, j, r]
    full = v.permute(0, 2, 1, 5, 6, 3, 4).reshape(nt * 8, kb * 128)
    return full[:n, :k].to(torch.uint8)
