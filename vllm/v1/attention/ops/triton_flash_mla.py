import torch
import triton
import triton.language as tl


@triton.jit
def _flash_mla_stage1(
    Q, K_Cache, V_Cache, Block_Table, Seq_Lens, Att_Out,
    # strides
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ko, stride_kd,
    stride_vb, stride_vo, stride_vd,
    stride_btb, stride_bts,
    stride_ab, stride_ah, stride_as, stride_ad,
    # config
    sm_scale: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_PE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_hg = tl.program_id(1)
    pid_split = tl.program_id(2)
    
    h_start = pid_hg * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    
    seq_len = tl.load(Seq_Lens + pid_b)

    split_size = tl.cdiv(seq_len, NUM_SPLITS)
    start = pid_split * split_size
    end = tl.minimum(start + split_size, seq_len)
    
    if end <= start:
        return  # empty split
    
    d_nope = tl.arange(0, D_NOPE)
    d_pe = D_NOPE + tl.arange(0, D_PE)
    
    q_base = Q + pid_b * stride_qb + h_offs[:, None] * stride_qh
    q_nope = tl.load(q_base + d_nope[None, :] * stride_qd).to(tl.float32)
    q_pe = tl.load(q_base + d_pe[None, :] * stride_qd).to(tl.float32)
    
    m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_NOPE], dtype=tl.float32)
    
    bt_base = Block_Table + pid_b * stride_btb
    
    # loop over blocks
    for k_start in range(start, end, BLOCK_N):
        k_offs = k_start + tl.arange(0, BLOCK_N)
        k_mask = k_offs < end
        
        page_ids = tl.load(bt_base + (k_offs // PAGE_SIZE) * stride_bts, mask=k_mask, other=0)
        page_offs = k_offs % PAGE_SIZE
        
        k_base = K_Cache + page_ids[:, None] * stride_kb + page_offs[:, None] * stride_ko
        
        k_nope = tl.load(k_base + d_nope[None, :] * stride_kd, mask=k_mask[:, None], other=0.0).to(tl.float32)
        k_pe = tl.load(k_base + d_pe[None, :] * stride_kd, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        qk = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_pe, tl.trans(k_pe))
        qk = qk * sm_scale
        qk = tl.where(k_mask[None, :], qk, float("-inf"))
        
        # online softmax
        m_new = tl.maximum(m, tl.max(qk, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        # load V nope
        v_base = V_Cache + page_ids[:, None] * stride_vb + page_offs[:, None] * stride_vo
        v = tl.load(v_base + d_nope[None, :] * stride_vd, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l = l * alpha + tl.sum(p, 1)
        m = m_new
    
    acc = acc / l[:, None]
    
    out_base = Att_Out + pid_b * stride_ab + h_offs[:, None] * stride_ah + pid_split * stride_as
    tl.store(out_base + d_nope[None, :] * stride_ad, acc)
    
    lse = m + tl.log(l)
    lse_base = Att_Out + pid_b * stride_ab + h_offs * stride_ah + pid_split * stride_as + D_NOPE * stride_ad
    tl.store(lse_base, lse)


@triton.jit
def _flash_mla_stage2(
    Att_Logits, Output, LSE, Seq_Lens,
    stride_ab, stride_ah, stride_as, stride_ad,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    D_NOPE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    seq_len = tl.load(Seq_Lens + pid_b)
    d_offs = tl.arange(0, D_NOPE)
    
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D_NOPE], dtype=tl.float32)
    
    for split in range(NUM_SPLITS):
        split_size = tl.cdiv(seq_len, NUM_SPLITS)
        split_start = split * split_size
        split_end = tl.minimum(split_start + split_size, seq_len)
        
        if split_end > split_start:
            base = Att_Logits + pid_b * stride_ab + pid_h * stride_ah + split * stride_as
            
            out = tl.load(base + d_offs * stride_ad)
            lse = tl.load(base + D_NOPE * stride_ad)
            
            m_new = tl.maximum(m, lse)
            alpha = tl.exp(m - m_new)
            w = tl.exp(lse - m_new)
            
            acc = acc * alpha + out * w
            l = l * alpha + w
            m = m_new
    
    result = acc / l
    
    out_ptr = Output + pid_b * stride_ob + pid_h * stride_oh
    tl.store(out_ptr + d_offs * stride_od, result.to(Output.dtype.element_ty))
    
    if RETURN_LSE:
        lse_val = m + tl.log(l)
        tl.store(LSE + pid_b * stride_lb + pid_h * stride_lh, lse_val)


def flash_mla_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    num_splits: int = 4,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:

    B, H, D_total = q.shape
    page_size = k_cache.shape[1]
    D_nope = v_cache.shape[-1]
    D_pe = D_total - D_nope
    
    device = q.device
    dtype = q.dtype
    
    att_logits = torch.empty(B, H, num_splits, D_nope + 1, dtype=torch.float32, device=device)
    output = torch.zeros(B, H, D_nope, dtype=dtype, device=device)
    lse = torch.zeros(B, H, dtype=torch.float32, device=device) if return_lse else None
    
    BLOCK_H = 16
    BLOCK_N = 32
    num_warps = 8
    num_stages = 2

    
    num_head_groups = triton.cdiv(H, BLOCK_H)
    
    # stage 1
    grid1 = (B, num_head_groups, num_splits)
    _flash_mla_stage1[grid1](
        q, k_cache, v_cache, block_table, seq_lens, att_logits,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(), *att_logits.stride(),
        sm_scale=sm_scale,
        PAGE_SIZE=page_size,
        D_NOPE=D_nope,
        D_PE=D_pe,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    # stage 2
    grid2 = (B, H)
    _flash_mla_stage2[grid2](
        att_logits, output, lse, seq_lens,
        *att_logits.stride(), *output.stride(),
        lse.stride(0) if return_lse else 0,
        lse.stride(1) if return_lse else 0,
        D_NOPE=D_nope,
        NUM_SPLITS=num_splits,
        RETURN_LSE=return_lse,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return output, lse