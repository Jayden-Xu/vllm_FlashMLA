import torch
import triton
from dataclasses import dataclass
from typing import ClassVar, Tuple, Optional, Any, Union

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
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# 导入算子
try:
    from vllm.v1.attention.ops.triton_flash_mla import (
        flash_mla_prefill_kernel,
        flash_mla_decode_stage_1_kernel,
        flash_mla_decode_stage_2_kernel
    )
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

logger = init_logger(__name__)

@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    max_seq_len: int

@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass

class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], 
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata)

    def _build_decode(self, block_table_tensor: torch.Tensor, seq_lens_device: torch.Tensor, 
                      max_seq_len: int, query_start_loc_cpu: torch.Tensor, 
                      query_start_loc_device: torch.Tensor, num_decode_tokens: int, 
                      dcp_tot_seq_lens_device: torch.Tensor | None) -> FlashMLADecodeMetadata:
        return FlashMLADecodeMetadata(
            block_table=block_table_tensor, 
            seq_lens=seq_lens_device, 
            max_seq_len=max_seq_len,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    def __init__(self, **mla_args) -> None:
        super().__init__(**mla_args)
        self.kv_lora_rank = mla_args.get("kv_lora_rank", 512)
        self.qk_rope_head_dim = mla_args.get("qk_rope_head_dim", 64)
        logger.info(f"⚡ [FlashMLA] Backend Loaded (Latent={self.kv_lora_rank}, RoPE={self.qk_rope_head_dim})")

    def _run_prefill_triton_impl(self, q, k, v, attn_metadata, kv_cache=None, **kwargs):
        q_nope, q_pe = q if isinstance(q, tuple) else (q[..., :512], q[..., 512:])
        k_final = torch.cat([v, k], dim=-1)

        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            slot_mapping = attn_metadata.slot_mapping.flatten()
            kv_cache.view(-1, kv_cache.shape[-1])[slot_mapping] = k_final.view(-1, k_final.shape[-1])

        output = torch.empty((q_nope.shape[0], self.num_heads, self.kv_lora_rank), 
                             dtype=q_nope.dtype, device=q_nope.device)
        
        if HAS_KERNELS:
            q_start_loc = attn_metadata.query_start_loc
            grid = lambda META: (triton.cdiv(attn_metadata.max_query_len, META['BLOCK_M']), q_start_loc.shape[0] - 1, self.num_heads)
            
            flash_mla_prefill_kernel[grid](
                q_nope, q_pe, k_final, q_start_loc, output,
                *q_nope.stride(), *q_pe.stride(), 
                k_final.stride(0), k_final.stride(2), 
                *output.stride(),
                self.scale, D_LATENT=512, D_ROPE=64
            )
        return output

    def _forward_decode(self, q, kv_cache, attn_metadata, layer=None):
        q_nope, q_pe = q
        decode_meta = attn_metadata.decode
        batch_size = q_nope.shape[0]
        kv_block_size = attn_metadata.kv_cache_spec.block_size

        output = torch.empty((batch_size, self.num_heads, self.kv_lora_rank), 
                             dtype=q_nope.dtype, device=q_nope.device)
        
        max_seq_len = decode_meta.max_seq_len
        # 启发式估算 Buffer 尺寸
        MAX_POSSIBLE_SPLITS = (max_seq_len + 127) // 128
        num_splits = max(1, min(128, 1 << (MAX_POSSIBLE_SPLITS - 1).bit_length() if MAX_POSSIBLE_SPLITS > 1 else 1))

        mid_o = torch.zeros((batch_size, self.num_heads, num_splits, self.kv_lora_rank), 
                            dtype=torch.float32, device=q_nope.device)
        mid_lse = torch.full((batch_size, self.num_heads, num_splits), float("-inf"), 
                             dtype=torch.float32, device=q_nope.device)

        if HAS_KERNELS:
            # Stage 1
            flash_mla_decode_stage_1_kernel[(batch_size, self.num_heads, num_splits)](
                q_nope, q_pe, kv_cache, decode_meta.block_table, decode_meta.seq_lens, mid_o, mid_lse,
                *q_nope.stride(), *q_pe.stride(), *kv_cache.stride(), *decode_meta.block_table.stride(),
                *mid_o.stride(), *mid_lse.stride(),
                self.scale, KV_BLOCK_SIZE=kv_block_size, D_LATENT=512, D_ROPE=64
            )

            # Stage 2
            flash_mla_decode_stage_2_kernel[(batch_size, self.num_heads)](
                mid_o, mid_lse, output,
                *mid_o.stride(), *mid_lse.stride(), *output.stride(),
                D_LATENT=512, NUM_SPLITS=num_splits
            )

        return output, None

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
        import os
        if os.getenv("DISABLE_FLASH_MLA") == "1":
            return False
        return capability.major >= 8