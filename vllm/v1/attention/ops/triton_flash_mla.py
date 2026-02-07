import triton
import triton.language as tl


@triton.heuristics({
    'BLOCK_N': lambda kwargs: 32,
    'num_warps': lambda kwargs: 4,
    'num_stages': lambda kwargs: 3,
})
@triton.jit
def flash_mla_decode_stage_1_kernel(
    Q_nope_ptr, Q_pe_ptr, KV_Cache_ptr, Block_Table_ptr, Seq_Lens_ptr, Mid_O_ptr, Mid_LSE_ptr,
    stride_qnb, stride_qnh, stride_qnd,
    stride_qpb, stride_qph, stride_qpd,
    stride_kvb, stride_kvo, stride_kvd,
    stride_btb, stride_bts,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_lb, stride_mid_lh, stride_mid_ls,
    sm_scale,
    SPLIT_SIZE, 
    KV_BLOCK_SIZE: tl.constexpr,
    D_LATENT: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_d_latent = tl.arange(0, D_LATENT)
    offs_d_rope = tl.arange(0, D_ROPE)
    
    # 1. 指针与加载 Q (转换为 2D 以便使用 tl.dot)
    qn = tl.load(Q_nope_ptr + pid_b * stride_qnb + pid_h * stride_qnh + offs_d_latent * stride_qnd).to(tl.float16)[None, :]
    qp = tl.load(Q_pe_ptr + pid_b * stride_qpb + pid_h * stride_qph + offs_d_rope * stride_qpd).to(tl.float16)[None, :]

    seq_len = tl.load(Seq_Lens_ptr + pid_b)
    start_n = pid_s * SPLIT_SIZE
    end_n = tl.minimum(start_n + SPLIT_SIZE, seq_len)

    # 边界检查
    if start_n >= seq_len:
        mid_lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lb + pid_h * stride_mid_lh + pid_s * stride_mid_ls
        tl.store(mid_lse_ptr, float("-inf"))
        return

    # 2. 初始化 (仿照 fused 版，强制使用 Tensor)
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([1, D_LATENT], dtype=tl.float32)

    num_valid_tokens = end_n - start_n
    num_full_blocks = num_valid_tokens // BLOCK_N
    limit = start_n + num_full_blocks * BLOCK_N
    
    bt_ptr_base = Block_Table_ptr + pid_b * stride_btb

    # 3. 主循环：处理完整的 BLOCK_N (触发 Tensor Core)
    for curr_n in range(start_n, limit, BLOCK_N):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        
        l_block_id = offs_n // KV_BLOCK_SIZE
        p_block_id = tl.load(bt_ptr_base + l_block_id * stride_bts)
        block_off = offs_n % KV_BLOCK_SIZE

        kv_base = KV_Cache_ptr + p_block_id[:, None] * stride_kvb + block_off[:, None] * stride_kvo
        kn = tl.load(kv_base + offs_d_latent[None, :] * stride_kvd).to(tl.float16)
        kp = tl.load(kv_base + (D_LATENT + offs_d_rope[None, :]) * stride_kvd).to(tl.float16)
        
        # 使用 tl.dot 代替 tl.sum(qn * kn)
        qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qp, tl.trans(kp))) * sm_scale

        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha + tl.dot(p.to(tl.float16), kn)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # 4. 尾部处理 (Peeling 部分)
    if limit < end_n:
        curr_n = limit
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end_n
        
        l_block_id = offs_n // KV_BLOCK_SIZE
        p_block_id = tl.load(bt_ptr_base + l_block_id * stride_bts, mask=mask_n, other=0)
        block_off = offs_n % KV_BLOCK_SIZE

        kv_base = KV_Cache_ptr + p_block_id[:, None] * stride_kvb + block_off[:, None] * stride_kvo
        kn = tl.load(kv_base + offs_d_latent[None, :] * stride_kvd, mask=mask_n[:, None], other=0.0).to(tl.float16)
        kp = tl.load(kv_base + (D_LATENT + offs_d_rope[None, :]) * stride_kvd, mask=mask_n[:, None], other=0.0).to(tl.float16)
        
        qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qp, tl.trans(kp))) * sm_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha + tl.dot(p.to(tl.float16), kn)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # 5. 归一化与写回
    inv_l = 1.0 / tl.maximum(l_i, 1e-10)
    acc_flat = tl.reshape(acc * inv_l, [D_LATENT])
    
    # 指针计算
    mid_o_ptr = Mid_O_ptr + pid_b * stride_mid_ob + pid_h * stride_mid_oh + pid_s * stride_mid_os + offs_d_latent * stride_mid_od
    mid_lse_ptr = Mid_LSE_ptr + pid_b * stride_mid_lb + pid_h * stride_mid_lh + pid_s * stride_mid_ls
    
    tl.store(mid_o_ptr, acc_flat.to(Mid_O_ptr.dtype.element_ty))
    # LSE 写回，l_i 已经是 Tensor，tl.log 会自动处理
    lse_val = tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf"))
    tl.store(mid_lse_ptr, lse_val)

@triton.heuristics({
    'num_warps': lambda kwargs: 8,
    'num_stages': lambda kwargs: 2,
})
@triton.jit
def flash_mla_decode_stage_2_kernel(
    Mid_O_ptr, Mid_LSE_ptr, Output_ptr,
    stride_mob, stride_moh, stride_mos, stride_mod,
    stride_mlb, stride_mlh, stride_mls,
    stride_ob, stride_oh, stride_od,
    NUM_SPLITS,
    D_LATENT: tl.constexpr, 
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # max split is 128 in backend
    offs_s = tl.arange(0, 128)
    mask_s = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, D_LATENT)
    
    lse = tl.load(Mid_LSE_ptr + pid_b * stride_mlb + pid_h * stride_mlh + offs_s * stride_mls, mask=mask_s, other=float("-inf"))
    m_global = tl.max(lse, 0)
    
    if m_global == float("-inf"):
        tl.store(Output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, 0.0)
        return

    weights = tl.exp(lse - m_global)
    weights = tl.where(mask_s & (lse > float("-inf")), weights, 0.0)
    sum_w = tl.sum(weights, 0)
    
    mid_o_ptrs = Mid_O_ptr + pid_b * stride_mob + pid_h * stride_moh + offs_s[:, None] * stride_mos + offs_d[None, :] * stride_mod
    mid_o = tl.load(mid_o_ptrs, mask=mask_s[:, None], other=0.0)
    
    final_o = tl.sum(mid_o * weights[:, None], 0) / (sum_w + 1e-10)
    tl.store(Output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, final_o.to(Output_ptr.dtype.element_ty))


@triton.heuristics({
    'BLOCK_N': lambda kwargs: 32,
    'num_warps': lambda kwargs: 4,
    'num_stages': lambda kwargs: 3,
})
@triton.jit
def flash_mla_decode_fused_kernel(
    Q_nope_ptr, Q_pe_ptr, KV_Cache_ptr, Block_Table_ptr, Seq_Lens_ptr, Output_ptr,
    stride_qnb, stride_qnh, stride_qnd,
    stride_qpb, stride_qph, stride_qpd,
    stride_kvb, stride_kvo, stride_kvd,
    stride_btb, stride_bts,
    stride_ob, stride_oh, stride_od,
    sm_scale,
    KV_BLOCK_SIZE: tl.constexpr,
    D_LATENT: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # 指针预计算
    bt_ptr_base = Block_Table_ptr + pid_b * stride_btb
    qn_ptr = Q_nope_ptr + pid_b * stride_qnb + pid_h * stride_qnh
    qp_ptr = Q_pe_ptr + pid_b * stride_qpb + pid_h * stride_qph
    
    offs_d_latent = tl.arange(0, D_LATENT)
    offs_d_rope = tl.arange(0, D_ROPE)
    
    qn = tl.load(qn_ptr + offs_d_latent * stride_qnd).to(tl.float16)[None, :]
    qp = tl.load(qp_ptr + offs_d_rope * stride_qpd).to(tl.float16)[None, :]

    seq_len = tl.load(Seq_Lens_ptr + pid_b)
    
    # --- 关键修改 1: 使用标准的 -inf ---
    # --- 关键修复：强行定义为 [1] 维张量，确保类型一致性 ---
    # 使用 tl.full 或 tl.zeros([1]) 确保它们从一开始就是 Tensor 而不是 Scalar
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([1, D_LATENT], dtype=tl.float32)

    # --- 关键修改 2: 还原 Loop Peeling (保持不变) ---
    full_blocks = (seq_len // BLOCK_N) * BLOCK_N

    # 1. 主循环：处理完整的 Block
    for start_n in range(0, full_blocks, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        l_block_id = offs_n // KV_BLOCK_SIZE
        p_block_id = tl.load(bt_ptr_base + l_block_id * stride_bts)
        block_off = offs_n % KV_BLOCK_SIZE

        kv_off = p_block_id[:, None] * stride_kvb + block_off[:, None] * stride_kvo
        kn = tl.load(KV_Cache_ptr + kv_off + offs_d_latent[None, :] * stride_kvd).to(tl.float16)
        kp = tl.load(KV_Cache_ptr + kv_off + (D_LATENT + offs_d_rope[None, :]) * stride_kvd).to(tl.float16)
        
        qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qp, tl.trans(kp))) * sm_scale
        
        # 现在 m_curr 是 [1]，m_i 也是 [1]，编译器就开心了
        m_curr = tl.max(qk, 1) 
        m_new = tl.maximum(m_i, m_curr)
        
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha + tl.dot(p.to(tl.float16), kn)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # 2. 尾部处理 (Peeling 出来的部分)
    # ... (逻辑同上，同样确保更新的是 [1] 维 Tensor) ...
    # 记得这里的 m_i 和 l_i 也会被更新
    # 2. 尾部处理 (Peeling 出来的部分)：处理剩余的 token
    if full_blocks < seq_len:
        start_n = full_blocks
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len # 关键：这里需要掩码
        
        # 加载逻辑
        l_block_id = offs_n // KV_BLOCK_SIZE
        p_block_id = tl.load(bt_ptr_base + l_block_id * stride_bts, mask=mask_n, other=0)
        block_off = offs_n % KV_BLOCK_SIZE

        kv_off = p_block_id[:, None] * stride_kvb + block_off[:, None] * stride_kvo
        
        # 注意：load 也要加 mask，否则会越界读取
        kn = tl.load(KV_Cache_ptr + kv_off + offs_d_latent[None, :] * stride_kvd, 
                     mask=mask_n[:, None], other=0.0).to(tl.float16)
        kp = tl.load(KV_Cache_ptr + kv_off + (D_LATENT + offs_d_rope[None, :]) * stride_kvd, 
                     mask=mask_n[:, None], other=0.0).to(tl.float16)
        
        # 计算 Score
        qk = (tl.dot(qn, tl.trans(kn)) + tl.dot(qp, tl.trans(kp))) * sm_scale
        
        # --- 关键：使用 float("-inf") 填充无效位置 ---
        # 确保 mask_n 被广播到 [1, BLOCK_N]
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Online Softmax 更新 (m_i 已经是 [1] 维 Tensor)
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)

        # 累加
        acc = acc * alpha + tl.dot(p.to(tl.float16), kn)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # --- 最终写回 ---
    acc = acc / tl.maximum(l_i, 1e-10)
    
    # 关键修复：将 [1, 512] 的 Rank 强制压平为 [512]
    # 这样 acc_flat 的 rank 就跟 out_ptr 一致了
    acc_flat = tl.reshape(acc, [D_LATENT])
    
    out_ptr = Output_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d_latent * stride_od
    tl.store(out_ptr, acc_flat.to(Output_ptr.dtype.element_ty))