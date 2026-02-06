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

try:
    from vllm.v1.attention.ops.triton_flash_mla import (
        flash_mla_decode_stage_1_kernel,
        flash_mla_decode_stage_2_kernel,
        flash_mla_decode_fused_kernel
    )
    HAS_DECODE_KERNELS = True
except ImportError:
    HAS_DECODE_KERNELS = False

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
    _printed_configs = set()
    _backend_logged = False
    _log_fused_once = True 
    _log_splitk_once = True
    

    def __init__(self, **mla_args) -> None:
        super().__init__(**mla_args)
        self.scale = mla_args.get("scale", 1.0)
        self.qk_rope_head_dim = mla_args.get("qk_rope_head_dim", 64)
        
        if not FlashMLAImpl._backend_logged:
            logger.info(f"[FlashMLA] Backend Hijacked (scale={self.scale:.4f}, rope_dim={self.qk_rope_head_dim})")
            FlashMLAImpl._backend_logged = True


    def _forward_decode(self, q, kv_cache, attn_metadata, layer=None):

        q_nope, q_pe = q
        decode_meta = attn_metadata.decode
        batch_size = q_nope.shape[0]
        actual_latent_dim = q_nope.shape[-1]
        max_seq_len = decode_meta.max_seq_len
        device = q_nope.device
        
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        base_splits = (max_seq_len + 255) // 256
        
        if batch_size >= 32:
            num_splits = 1
        else:
            if batch_size <= 4:
                num_splits = min(base_splits, 64)
            elif batch_size * base_splits <= num_sms * 4:
                num_splits = base_splits
            else:
                num_splits = max(1, (num_sms * 4) // batch_size)
            
            if num_splits > 1:
                num_splits = 1 << (num_splits - 1).bit_length()
            num_splits = min(num_splits, 128)

        
        output = torch.empty((batch_size, self.num_heads, actual_latent_dim), 
                             dtype=q_nope.dtype, device=device)

        if num_splits == 1:
            flash_mla_decode_fused_kernel[(batch_size, self.num_heads)](
                q_nope, q_pe, kv_cache, 
                decode_meta.block_table, decode_meta.seq_lens, 
                output,
                *q_nope.stride(), 
                *q_pe.stride(), 
                *kv_cache.stride(), 
                *decode_meta.block_table.stride(),
                *output.stride(),
                self.scale, 
                KV_BLOCK_SIZE=kv_cache.shape[-2], 
                D_LATENT=actual_latent_dim, 
                D_ROPE=self.qk_rope_head_dim,
            )
            
            if FlashMLAImpl._log_fused_once and hasattr(flash_mla_decode_fused_kernel, 'best_config'):
                self._print_best_config("Fused Path", flash_mla_decode_fused_kernel)
                FlashMLAImpl._log_fused_once = False

        else:
            actual_split_size = (max_seq_len + num_splits - 1) // num_splits
            
            mid_o = torch.empty((batch_size, self.num_heads, num_splits, actual_latent_dim), 
                                dtype=torch.float32, device=device)
            mid_lse = torch.empty((batch_size, self.num_heads, num_splits), 
                                 dtype=torch.float32, device=device)

            flash_mla_decode_stage_1_kernel[(batch_size, self.num_heads, num_splits)](
                q_nope, q_pe, kv_cache, decode_meta.block_table, decode_meta.seq_lens, 
                mid_o, mid_lse,
                *q_nope.stride(), *q_pe.stride(), *kv_cache.stride(), 
                *decode_meta.block_table.stride(), *mid_o.stride(), *mid_lse.stride(),
                self.scale, int(actual_split_size), 
                KV_BLOCK_SIZE=kv_cache.shape[-2], D_LATENT=actual_latent_dim, D_ROPE=self.qk_rope_head_dim,
            )

            flash_mla_decode_stage_2_kernel[(batch_size, self.num_heads)](
                mid_o, mid_lse, output,
                *mid_o.stride(), *mid_lse.stride(), *output.stride(),
                int(num_splits), D_LATENT=actual_latent_dim, 
            )

            if FlashMLAImpl._log_splitk_once and hasattr(flash_mla_decode_stage_1_kernel, 'best_config'):
                self._print_best_config("Split-K Stage 1", flash_mla_decode_stage_1_kernel)
                self._print_best_config("Split-K Stage 2", flash_mla_decode_stage_2_kernel)
                FlashMLAImpl._log_splitk_once = False

        return output, None


    def _print_best_config(self, name, kernel):
        if hasattr(kernel, 'best_config'):
            cfg = kernel.best_config
            parts = [f"warps={cfg.num_warps}", f"stages={cfg.num_stages}"]
            if hasattr(cfg, 'kwargs') and cfg.kwargs:
                for k, v in cfg.kwargs.items():
                    parts.append(f"{k}={v}")
            config_str = "{" + ", ".join(parts) + "}"
            logger.info(f"[FlashMLA] {name}: {config_str}")


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