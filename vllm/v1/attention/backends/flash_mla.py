
import torch, triton
from dataclasses import dataclass
from typing import ClassVar, Tuple, List, Type
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

logger = init_logger(__name__)

# import FlashMLA kernels
try:
    from vllm.v1.attention.ops import (
        flash_mla_decode_kernel,
        flash_mla_prefill_kernel
    )
    HAS_KERNELS = True
except ImportError:
    logger.warning("⚠️ [FlashMLA] Kernels not found! Running in structure-verification mode.")
    HAS_KERNELS = False
    flash_mla_decode_kernel = None
    flash_mla_prefill_kernel = None



class FlashMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

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
        # unlock support for compute capability 8
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

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        self._run_prefill_new_tokens = self._run_prefill_triton_impl
        self._run_prefill_context_chunk = self._run_prefill_chunk_triton_impl

        self.kv_lora_rank = self.mla_dims.kv_lora_rank # 512
        self.qk_rope_head_dim = self.mla_dims.qk_rope_head_dim # 64

        logger.info(f"⚡️ [TritonMLA] Initialized on {torch.cuda.get_device_name()}! Mode: Triton Pure Python")

    # prefill (ragged)
    def _run_prefill_triton_impl(
        self,
        prefill_metadata: MLACommonMetadata,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool
    ):
        """
        q: [Total, Heads, 576] (Latent_Q + RoPE_Q)
        k: [Total, 1, 576] (Latent_K + RoPE_K)
        v: [Total, 1, 512] (Latent_V = Latent_K
        """

        cu_seqlens = prefill_metadata.query_start_loc
        num_seqs = cu_seqlens.shape[0] - 1
        max_seq_len = prefill_metadata.max_query_len

        output = torch.empty_like(v)

        if not HAS_KERNELS:
            return output, None
        
        BLOCK_M = 64
        grid = (triton.cdiv(max_seq_len, BLOCK_M), num_seqs, self.num_heads)

        flash_mla_prefill_kernel[grid](
            Q_ptr = q,
            KV_ptr = k, # Latent + RoPE
            cu_seqlens_ptr = cu_seqlens,
            Output_ptr = output,
            # strides
            stride_q_n=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_kv_n=k.stride(0), stride_kv_h=k.stride(1), stride_kv_d=k.stride(2),

            sm_scale = self.scale,
            D_LATENT = self.kv_lora_rank,
            D_ROPE = self.qk_rope_head_dim,

            BLOCK_M = BLOCK_M,
            BLOCK_N = 32
        )

        return output, None
    

    def _run_prefill_chunk_triton_impl(
        self, *args, **kwargs
    ):
        raise NotImplementedError("Triton MLA Chunked Prefill not implemented yet.")


    # decode (paged)
    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # prepare metadata
        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens

        output = torch.empty_like(q_nope)
        
        output.fill_(0)

        # kernel calls
        # flash_mla_decode_kenrel[grid](...)

        return output, None
       
