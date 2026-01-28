import triton
import triton.language as tl


def get_prefill_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ]

def get_decode_stage1_configs():
    configs = []
    for split_size in [128, 256]:
        for block_n in [16, 32, 64]:
            if block_n <= split_size:
                 configs.append(triton.Config({'BLOCK_N': block_n, 'SPLIT_SIZE': split_size}, num_warps=4, num_stages=2))
    for split_size in [512, 1024]:
        for block_n in [64, 128, 256]:
             if block_n <= split_size:
                configs.append(triton.Config({'BLOCK_N': block_n, 'SPLIT_SIZE': split_size}, num_warps=8, num_stages=3))
    return configs

def get_decode_stage2_configs():
    return [
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]


@triton.autotune(configs=get_prefill_configs(), key=['D_LATENT', 'D_ROPE'])
@triton.jit
def flash_mla_prefill_kernel(
    Q_nope_ptr, Q_pe_ptr, KV_ptr, cu_seqlens_ptr, Output_ptr,
    stride_qnn, stride_qnh, stride_qnd,
    stride_qpn, stride_qph, stride_qpd,
    stride_kvn, stride_kvd,
    stride_on, stride_oh, stride_od,
    sm_scale,
    D_LATENT: tl.constexpr, D_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_b = tl.program_id(1); pid_h = tl.program_id(2)
    
    start_q = tl.load(cu_seqlens_ptr + pid_b)
    end_q = tl.load(cu_seqlens_ptr + pid_b + 1)
    seq_len = end_q - start_q
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_latent = tl.arange(0, D_LATENT)
    offs_d_rope = tl.arange(0, D_ROPE)
    
    q_n_ptrs = Q_nope_ptr + (start_q + offs_m[:, None]) * stride_qnn + pid_h * stride_qnh + offs_d_latent[None, :] * stride_qnd
    q_p_ptrs = Q_pe_ptr + (start_q + offs_m[:, None]) * stride_qpn + pid_h * stride_qph + offs_d_rope[None, :] * stride_qpd
    
    qn = tl.load(q_n_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
    qp = tl.load(q_p_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_LATENT], dtype=tl.float32)

    limit_n = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    
    for start_n in range(0, limit_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        kv_base = KV_ptr + (start_q + offs_n[None, :]) * stride_kvn
        kn_ptrs = kv_base + offs_d_latent[:, None] * stride_kvd
        kp_ptrs = kv_base + (D_LATENT + offs_d_rope[:, None]) * stride_kvd
        
        kn = tl.load(kn_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)
        kp = tl.load(kp_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)
        
        qk = (tl.dot(qn.to(tl.float16), kn.to(tl.float16)) + 
              tl.dot(qp.to(tl.float16), kp.to(tl.float16))) * sm_scale

        if start_n + BLOCK_N > pid_m * BLOCK_M:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        
        # online Softmax
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), tl.trans(kn).to(tl.float16), acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = Output_ptr + (start_q + offs_m[:, None]) * stride_on + pid_h * stride_oh + offs_d_latent[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Output_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_len)


@triton.autotune(configs=get_decode_stage1_configs(), key=['D_LATENT', 'D_ROPE', 'KV_BLOCK_SIZE'])
@triton.jit
def flash_mla_decode_stage_1_kernel(
    Q_nope_ptr, Q_pe_ptr, KV_Cache_ptr, Block_Table_ptr, Seq_Lens_ptr, Mid_O_ptr, Mid_LSE_ptr,
    stride_qnb, stride_qnh, stride_qnd,
    stride_qpb, stride_qph, stride_qpd,
    stride_kvb, stride_kvo, stride_kvd,
    stride_btb, stride_bts,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_lb, stride_mid_lh, stride_mid_ls,
    sm_scale, KV_BLOCK_SIZE: tl.constexpr, D_LATENT: tl.constexpr, D_ROPE: tl.constexpr, 
    BLOCK_N: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0); pid_h = tl.program_id(1); pid_s = tl.program_id(2)
    
    offs_d_latent = tl.arange(0, D_LATENT)
    offs_d_rope = tl.arange(0, D_ROPE)
    
    qn = tl.load(Q_nope_ptr + pid_b * stride_qnb + pid_h * stride_qnh + offs_d_latent * stride_qnd).to(tl.float32)
    qp = tl.load(Q_pe_ptr + pid_b * stride_qpb + pid_h * stride_qph + offs_d_rope * stride_qpd).to(tl.float32)

    seq_len = tl.load(Seq_Lens_ptr + pid_b)
    start_n = pid_s * SPLIT_SIZE
    end_n = tl.minimum(start_n + SPLIT_SIZE, seq_len)

    if start_n >= seq_len:
        tl.store(Mid_LSE_ptr + pid_b * stride_mid_lb + pid_h * stride_mid_lh + pid_s * stride_mid_ls, float("-inf"))
        return

    m_i = float("-inf"); l_i = 0.0
    acc = tl.zeros([D_LATENT], dtype=tl.float32)

    for curr_n in range(start_n, end_n, BLOCK_N):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end_n
        
        l_block_id = offs_n // KV_BLOCK_SIZE
        block_off = offs_n % KV_BLOCK_SIZE
        p_block_id = tl.load(Block_Table_ptr + pid_b * stride_btb + l_block_id * stride_bts, mask=mask_n, other=0)

        kv_base = KV_Cache_ptr + p_block_id[:, None] * stride_kvb + block_off[:, None] * stride_kvo
        kn = tl.load(kv_base + offs_d_latent[None, :] * stride_kvd, mask=mask_n[:, None], other=0.0)
        kp = tl.load(kv_base + (D_LATENT + offs_d_rope[None, :]) * stride_kvd, mask=mask_n[:, None], other=0.0)
        
        score = (tl.sum(qn[None, :] * kn, 1) + tl.sum(qp[None, :] * kp, 1)) * sm_scale
        score = tl.where(mask_n, score, float("-inf"))

        m_curr = tl.max(score, 0)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(score - m_new)
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha + tl.sum(kn * p[:, None], 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new

    if l_i > 0:
        tl.store(Mid_O_ptr + pid_b * stride_mid_ob + pid_h * stride_mid_oh + pid_s * stride_mid_os + offs_d_latent * stride_mid_od, acc / l_i)
        tl.store(Mid_LSE_ptr + pid_b * stride_mid_lb + pid_h * stride_mid_lh + pid_s * stride_mid_ls, m_i + tl.log(l_i))


@triton.autotune(configs=get_decode_stage2_configs(), key=['D_LATENT', 'NUM_SPLITS'])
@triton.jit
def flash_mla_decode_stage_2_kernel(
    Mid_O_ptr, Mid_LSE_ptr, Output_ptr,
    stride_mob, stride_moh, stride_mos, stride_mod,
    stride_mlb, stride_mlh, stride_mls,
    stride_ob, stride_oh, stride_od,
    D_LATENT: tl.constexpr, NUM_SPLITS: tl.constexpr
):
    pid_b = tl.program_id(0); pid_h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D_LATENT)
    
    lse = tl.load(Mid_LSE_ptr + pid_b * stride_mlb + pid_h * stride_mlh + offs_s * stride_mls)
    m_global = tl.max(lse, 0)
    
    if m_global == float("-inf"):
        tl.store(Output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, 0.0)
        return

    weights = tl.exp(lse - m_global)
    weights = tl.where(lse > float("-inf"), weights, 0.0)
    sum_w = tl.sum(weights, 0)
    
    mid_o_ptrs = Mid_O_ptr + pid_b * stride_mob + pid_h * stride_moh + offs_s[:, None] * stride_mos + offs_d[None, :] * stride_mod
    mid_o = tl.load(mid_o_ptrs, mask=lse[:, None] > float("-inf"), other=0.0)
    
    final_o = tl.sum(mid_o * weights[:, None], 0) / sum_w
    tl.store(Output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, final_o.to(Output_ptr.dtype.element_ty))