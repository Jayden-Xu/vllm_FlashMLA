import torch
import triton
from dataclasses import dataclass
from typing import ClassVar, Tuple, List, Type, Optional

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

try:
    from vllm.v1.attention.ops.triton_flash_mla import (
        flash_mla_prefill_kernel,
        flash_mla_decode_stage_1_kernel,
        flash_mla_decode_stage_2_kernel
    )
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False
    flash_mla_prefill_kernel = None
    flash_mla_decode_stage_1_kernel = None
    flash_mla_decode_stage_2_kernel = None

logger = init_logger(__name__)

class FlashMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_MLA"

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8

@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    pass

@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass

class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata)

    def _build_decode(self, block_table_tensor: torch.Tensor, seq_lens_device: torch.Tensor, max_seq_len: int, query_start_loc_cpu: torch.Tensor, query_start_loc_device: torch.Tensor, num_decode_tokens: int, dcp_tot_seq_lens_device: torch.Tensor | None) -> FlashMLADecodeMetadata:
        return FlashMLADecodeMetadata(block_table=block_table_tensor, seq_lens=seq_lens_device, dcp_tot_seq_lens=dcp_tot_seq_lens_device)

class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes: list[float] | None, sliding_window: int | None, kv_cache_dtype: str, logits_soft_cap: float | None, attn_type: str, kv_sharing_target_layer_name: str | None, **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, **mla_args)
        self._run_prefill_new_tokens = self._run_prefill_triton_impl
        self._run_prefill_context_chunk = self._run_prefill_chunk_triton_impl
        self.kv_lora_rank = mla_args.get("kv_lora_rank", 512)
        self.qk_rope_head_dim = mla_args.get("qk_rope_head_dim", 64)
        if not HAS_KERNELS:
            logger.warning("⚠️ [FlashMLA] Kernels not found!")
        else:
            logger.info(f"⚡️ [TritonMLA] Initialized on {torch.cuda.get_device_name()}! Mode: Triton Pure Python")

    def forward(self, *args, **kwargs):
        self.current_metadata = kwargs.get("attn_metadata")
        if self.current_metadata is None:
            for arg in args:
                if hasattr(arg, "query_start_loc") or hasattr(arg, "decode_metadata") or hasattr(arg, "decode"):
                    self.current_metadata = arg; break
        return super().forward(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Prefill (Simple Tensor Output)
    # -------------------------------------------------------------------------
    def _run_prefill_triton_impl(self, *args, **kwargs):
        attn_metadata = getattr(self, "current_metadata", None)
        if attn_metadata is None:
            attn_metadata = kwargs.get("attn_metadata")
            if attn_metadata is None:
                for arg in args:
                    if hasattr(arg, "query_start_loc"): attn_metadata = arg; break
        
        q_raw = kwargs.get('q') if 'q' in kwargs else args[0]
        k_raw = kwargs.get('k') if 'k' in kwargs else args[1]
        v_raw = kwargs.get('v') if 'v' in kwargs else args[2]

        output = torch.empty((q_raw.shape[0], self.num_heads, self.kv_lora_rank), dtype=q_raw.dtype, device=q_raw.device)

        if not HAS_KERNELS or attn_metadata is None:
            return output

        cu_seqlens = attn_metadata.query_start_loc
        num_seqs = cu_seqlens.shape[0] - 1
        max_seq_len = attn_metadata.max_query_len
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(max_seq_len, BLOCK_M), num_seqs, self.num_heads)

        # Prefill 保持拼接模式，参数较少
        flash_mla_prefill_kernel[grid](
            q_raw, k_raw, cu_seqlens, output,
            q_raw.stride(0), q_raw.stride(1), q_raw.stride(2),
            k_raw.stride(0), k_raw.stride(1), k_raw.stride(2),
            self.scale,
            D_LATENT=self.kv_lora_rank, D_ROPE=self.qk_rope_head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        return output

    def _run_prefill_chunk_triton_impl(self, *args, **kwargs):
        raise NotImplementedError("Triton MLA Chunked Prefill not implemented yet.")

    # -------------------------------------------------------------------------
    # Decode (Tuple Output) - 修复参数错位
    # -------------------------------------------------------------------------
    def _forward_decode(self, q: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: FlashMLAMetadata, layer: Optional[AttentionLayer] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. 拼接 Q
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q_raw = torch.cat([q_nope, q_pe], dim=-1)
        else:
            q_raw = q

        batch_size = q_raw.shape[0]
        output = torch.empty((batch_size, self.num_heads, self.kv_lora_rank), dtype=q_raw.dtype, device=q_raw.device)

        if not HAS_KERNELS:
            return output, None

        decode_meta = getattr(attn_metadata, "decode_metadata", None) or getattr(attn_metadata, "decode", None)
        block_table = decode_meta.block_table
        seq_lens = decode_meta.seq_lens
        
        # 静态计算 Grid (避免 CPU 同步)
        FIXED_MAX_SEQ_LEN = 4096 
        SPLIT_SIZE = 2048
        NUM_SPLITS = min(max(triton.cdiv(FIXED_MAX_SEQ_LEN, SPLIT_SIZE), 1), 128)

        mid_o = torch.empty((batch_size, self.num_heads, NUM_SPLITS, self.kv_lora_rank), dtype=torch.float32, device=q_raw.device)
        mid_lse = torch.empty((batch_size, self.num_heads, NUM_SPLITS), dtype=torch.float32, device=q_raw.device)

        if kv_c_and_k_pe_cache is not None:
            grid_stage_1 = (batch_size, self.num_heads, NUM_SPLITS)
            
            # 🔥 这里的参数顺序必须严格匹配 triton_mla_kernels.py 中的 flash_mla_decode_stage_1_kernel
            # 我们移除了 RoPE 指针、Cos/Sin 指针、CUR_POS、N_CTX 等多余参数
            flash_mla_decode_stage_1_kernel[grid_stage_1](
                q_raw,                  # Q_ptr (拼接后的)
                kv_c_and_k_pe_cache,    # KV_Cache_ptr (物理 Paged Memory)
                block_table,            # Block_Table_ptr
                seq_lens,               # Seq_Lens_ptr
                mid_o,                  # Mid_O_ptr
                mid_lse,                # Mid_LSE_ptr
                
                # Strides (Q)
                q_raw.stride(0), q_raw.stride(1), q_raw.stride(2),
                
                # Strides (KV Cache)
                kv_c_and_k_pe_cache.stride(0), kv_c_and_k_pe_cache.stride(1), kv_c_and_k_pe_cache.stride(2),
                
                # Strides (Block Table)
                block_table.stride(0), block_table.stride(1),
                
                # Strides (Mid O)
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
                
                # Strides (Mid LSE)
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                
                # Scalar (必须在 constexpr 之前)
                self.scale,             # sm_scale
                
                # Constexpr
                KV_BLOCK_SIZE=16,
                D_LATENT=self.kv_lora_rank,
                D_ROPE=self.qk_rope_head_dim,
                BLOCK_N=16,
                SPLIT_SIZE=SPLIT_SIZE
            )

            # Stage 2
            grid_stage_2 = (batch_size, self.num_heads)
            flash_mla_decode_stage_2_kernel[grid_stage_2](
                mid_o, mid_lse, output,
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                D_LATENT=self.kv_lora_rank,
                NUM_SPLITS=NUM_SPLITS
            )

        return output, None