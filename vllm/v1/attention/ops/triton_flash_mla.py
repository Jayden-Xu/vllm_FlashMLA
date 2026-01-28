import torch
import triton
import triton.language as tl

# ============================================================================
#  Flash MLA Prefill Kernel (Fixed for Non-Power-of-2 Head Dim)
# ============================================================================
@triton.jit
def flash_mla_prefill_kernel(
    Q_ptr, KV_ptr, cu_seqlens_ptr, Output_ptr,
    stride_q_n, stride_q_h, stride_q_d,
    stride_kv_n, stride_kv_h, stride_kv_d,
    sm_scale,
    D_LATENT: tl.constexpr, D_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_b = tl.program_id(1); pid_h = tl.program_id(2)
    
    # 512 + 64 = 576
    HEAD_DIM: tl.constexpr = D_LATENT + D_ROPE
    # Triton arange 必须是 2 的幂，所以我们用 1024，然后 mask 掉多余的
    PADDED_HEAD_DIM: tl.constexpr = 1024 

    cu_seqlens_start = tl.load(cu_seqlens_ptr + pid_b)
    cu_seqlens_end = tl.load(cu_seqlens_ptr + pid_b + 1)
    seq_len = cu_seqlens_end - cu_seqlens_start
    if pid_m * BLOCK_M >= seq_len: return
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    q_token_idx = cu_seqlens_start + offs_m
    mask_m = offs_m < seq_len
    
    # ----------------------------------------------------------------
    # 修正：使用 PADDED_HEAD_DIM 并 Mask
    # ----------------------------------------------------------------
    offs_d = tl.arange(0, PADDED_HEAD_DIM)
    mask_d = offs_d < HEAD_DIM # 只保留前 576 个元素
    
    q_ptrs = Q_ptr + q_token_idx[:, None] * stride_q_n + pid_h * stride_q_h + offs_d[None, :] * stride_q_d
    # Mask 需要同时考虑 Sequence 边界 (M) 和 维度边界 (D)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = (q * sm_scale).to(Q_ptr.dtype.element_ty)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, D_LATENT], dtype=tl.float32)
    
    limit_n = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    offs_d_latent = tl.arange(0, D_LATENT)
    
    for start_n in range(0, limit_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_token_idx = cu_seqlens_start + offs_n
        mask_n = offs_n < seq_len
        
        # Load K (576 -> 1024 Padded)
        # K 转置后: [1024, BLOCK_N]
        k_ptrs = KV_ptr + kv_token_idx[None, :] * stride_kv_n + offs_d[:, None] * stride_kv_d
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        
        # Load V (Only Latent 512, Power of 2, OK)

        v_ptrs = KV_ptr + kv_token_idx[:, None] * stride_kv_n + offs_d_latent[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Dot Product (q: [M, 1024], k: [1024, N])
        # 多出来的部分是 0 * 0 = 0，不影响结果
        qk = tl.dot(q, k)
        
        if start_n + BLOCK_N > start_m:
            mask_causal = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask_causal, qk, float("-inf"))
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
        else:
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)
        l_curr = tl.sum(p, 1)
        l_i = l_i * alpha + l_curr
        m_i = m_new
        
    acc = acc / l_i[:, None]
    out_ptrs = Output_ptr + q_token_idx[:, None] * stride_q_n + pid_h * stride_q_h + offs_d_latent[None, :] * stride_q_d
    tl.store(out_ptrs, acc.to(Output_ptr.dtype.element_ty), mask=mask_m[:, None])


# ============================================================================
#  Flash MLA Decode Kernel - Stage 1 (Fixed for Non-Power-of-2 Head Dim)
# ============================================================================
@triton.jit
def flash_mla_decode_stage_1_kernel(
    Q_ptr, KV_Cache_ptr, Block_Table_ptr, Seq_Lens_ptr,
    Mid_O_ptr, Mid_LSE_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_kv_block, stride_kv_offset, stride_kv_d,
    stride_bt_b, stride_bt_s,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_d,
    stride_mid_lse_b, stride_mid_lse_h, stride_mid_lse_s,
    sm_scale,
    KV_BLOCK_SIZE: tl.constexpr, 
    D_LATENT: tl.constexpr, 
    D_ROPE: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    SPLIT_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0); pid_h = tl.program_id(1); pid_s = tl.program_id(2)

    HEAD_DIM: tl.constexpr = D_LATENT + D_ROPE
    PADDED_HEAD_DIM: tl.constexpr = 1024 # Fix: Pad 576 to 1024

    # 1. Load Query
    offs_d = tl.arange(0, PADDED_HEAD_DIM)
    mask_d = offs_d < HEAD_DIM
    
    q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + offs_d * stride_q_d
    q = tl.load(q_ptr, mask=mask_d, other=0.0)
    q = (q * sm_scale).to(Q_ptr.dtype.element_ty)

    seq_len = tl.load(Seq_Lens_ptr + pid_b)
    start_n = pid_s * SPLIT_SIZE
    end_n = tl.minimum(start_n + SPLIT_SIZE, seq_len)

    if start_n >= seq_len:
        lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lse_b + pid_h * stride_mid_lse_h + pid_s * stride_mid_lse_s
        tl.store(lse_ptr, float("-inf"))
        return

    m_i = float("-inf"); l_i = 0.0
    acc = tl.zeros([D_LATENT], dtype=tl.float32)

    for curr_n in range(start_n, end_n, BLOCK_N):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end_n
        
        logical_block_ids = offs_n // KV_BLOCK_SIZE
        block_offsets = offs_n % KV_BLOCK_SIZE
        bt_ptrs = Block_Table_ptr + pid_b * stride_bt_b + logical_block_ids * stride_bt_s
        physical_block_ids = tl.load(bt_ptrs, mask=mask_n, other=0)

        # Load K (Transposed [1024, BLOCK_N])
        k_ptrs = KV_Cache_ptr + \
                 physical_block_ids[:, None] * stride_kv_block + \
                 block_offsets[:, None] * stride_kv_offset + \
                 offs_d[None, :] * stride_kv_d
        
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Compute Score [BLOCK_N]
        # q: [1024], k: [BLOCK_N, 1024]
        # broadcasting q to [1, 1024] * [BLOCK_N, 1024] -> sum(axis=1) -> [BLOCK_N]
        score = tl.sum(q[None, :] * k, 1)
        score = tl.where(mask_n, score, float("-inf"))

        m_curr = tl.max(score, 0)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)

        # Load V (Latent 512, OK)
        offs_d_latent = tl.arange(0, D_LATENT)
        v_ptrs = KV_Cache_ptr + \
                 physical_block_ids[:, None] * stride_kv_block + \
                 block_offsets[:, None] * stride_kv_offset + \
                 offs_d_latent[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        weighted_v = tl.sum(v * p[:, None], 0)
        acc = acc * alpha + weighted_v
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new

    if l_i > 0:
        mid_o = acc / l_i
        mid_lse = m_i + tl.log(l_i)
    else:
        mid_o = tl.zeros([D_LATENT], dtype=tl.float32)
        mid_lse = float("-inf")

    offs_d_latent = tl.arange(0, D_LATENT)
    o_ptr = Mid_O_ptr + pid_b * stride_mid_o_b + pid_h * stride_mid_o_h + \
            pid_s * stride_mid_o_s + offs_d_latent * stride_mid_o_d
    tl.store(o_ptr, mid_o)

    lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lse_b + pid_h * stride_mid_lse_h + pid_s * stride_mid_lse_s
    tl.store(lse_ptr, mid_lse)


# ============================================================================
#  Flash MLA Decode Kernel - Stage 2 (Keep this as is)
# ============================================================================
@triton.jit
def flash_mla_decode_stage_2_kernel(
    Mid_O_ptr, Mid_LSE_ptr, Output_ptr,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_d,
    stride_mid_lse_b, stride_mid_lse_h, stride_mid_lse_s,
    stride_out_b, stride_out_h, stride_out_d,
    D_LATENT: tl.constexpr, 
    NUM_SPLITS: tl.constexpr
):
    pid_b = tl.program_id(0); pid_h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D_LATENT)
    lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lse_b + pid_h * stride_mid_lse_h + offs_s * stride_mid_lse_s
    lse = tl.load(lse_ptr, mask=offs_s < NUM_SPLITS, other=float("-inf"))
    m_global = tl.max(lse, 0)
    if m_global == float("-inf"): weights = tl.zeros([NUM_SPLITS], dtype=tl.float32)
    else: weights = tl.exp(lse - m_global)
    sum_weights = tl.sum(weights, 0)
    mid_o_ptr = Mid_O_ptr + pid_b * stride_mid_o_b + pid_h * stride_mid_o_h + offs_s[:, None] * stride_mid_o_s + offs_d[None, :] * stride_mid_o_d
    mid_o = tl.load(mid_o_ptr, mask=offs_s[:, None] < NUM_SPLITS, other=0.0)
    weighted_o = mid_o * weights[:, None]
    final_o = tl.sum(weighted_o, 0)
    if sum_weights > 0: final_o = final_o / sum_weights
    out_ptr = Output_ptr + pid_b * stride_out_b + pid_h * stride_out_h + offs_d * stride_out_d
    tl.store(out_ptr, final_o.to(Output_ptr.dtype.element_ty))