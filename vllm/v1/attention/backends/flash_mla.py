import torch
from dataclasses import dataclass
from typing import ClassVar

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
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm.v1.attention.ops.triton_flash_mla import flash_mla_decode

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
    
    _logged_configs = set()
    
    def __init__(self, **mla_args) -> None:
        super().__init__(**mla_args)
        self.scale = mla_args.get("scale", 1.0)
        self.qk_rope_dim = mla_args.get("qk_rope_head_dim", 64)
        self.num_heads = mla_args.get("num_heads", 16)
        self.num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        
        logger.info(f"[FlashMLA] Initialized (Num Heads: {self.num_heads}, Scale: {self.scale}, QK RoPE dim: {self.qk_rope_dim}, Num SMs: {self.num_sms})")

    def _get_num_splits(self, batch_size: int) -> int:

        BLOCK_H = 8 
        heads_per_block = max(1, self.num_heads // BLOCK_H) 
        base_grid = batch_size * heads_per_block

        if base_grid == 0: return 1
        
        candidate_splits = [1, 2, 4, 8, 16, 32, 64]
        best_split = 1
        
        for s in candidate_splits:
            current_grid = base_grid * s
            
            # smallest split to feed SM
            if current_grid >= self.num_sms:
                # avoid tail effect
                waves = current_grid / self.num_sms
                
                if waves < 1.5:
                    # fallback to avoid tail effect
                    prev_s = s // 2 if s > 1 else 1
                    if (base_grid * prev_s) > (self.num_sms * 0.5):
                        return prev_s
                
                return s
            
            best_split = s
        
        return best_split


    def _forward_decode(self, q, kv_cache, attn_metadata, layer=None):
        decode_meta = attn_metadata.decode
        batch_size = q[0].shape[0] if isinstance(q, tuple) else q.shape[0]
        
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)
        
        d_nope = self.kv_lora_rank
        
        if kv_cache.ndim == 4:
            kv_cache = kv_cache.squeeze(2)
        
        k_cache = kv_cache
        v_cache = kv_cache[..., :d_nope]  # v only has nope
        
        # dynamic split-K
        num_splits = self._get_num_splits(batch_size)

        config_key = (batch_size, num_splits)
        if config_key not in FlashMLAImpl._logged_configs:
            logger.info(
                f"[FlashMLA-Config] NumSplits={num_splits}"
            )
            FlashMLAImpl._logged_configs.add(config_key)
        
        output, lse = flash_mla_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=decode_meta.block_table,
            seq_lens=decode_meta.seq_lens,
            sm_scale=self.scale,
            num_splits=num_splits,
            return_lse=True,
        )
        
        return output, lse


class FlashMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "FLASH_MLA"

    @staticmethod
    def get_impl_cls():
        return FlashMLAImpl

    @staticmethod
    def get_builder_cls():
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability):
        return capability >= DeviceCapability(8, 0)