"""Public API for the Chimera 2.1 MVP."""

from .bitlinear import BitLinear
from .bitconv import BitConv2d, ChimeraConv2d
from .logging_utils import make_benchmark_record
from .optimizer import (
    Chimera21,
    optimizer_state_memory_bytes,
    optimizer_state_memory_summary,
)
from .quantization import (
    abs_stat,
    quantize_weight_chimera,
    quantize_weight_strict_bitnet,
    ternary_stats,
)
from .target_bits import (
    SparseBranchController,
    TargetBitsController,
    entropy_from_ternary_ratios,
    symmetric_zero_ratio_for_entropy,
)

__all__ = [
    "BitLinear",
    "BitConv2d",
    "ChimeraConv2d",
    "Chimera21",
    "optimizer_state_memory_bytes",
    "optimizer_state_memory_summary",
    "abs_stat",
    "make_benchmark_record",
    "quantize_weight_chimera",
    "quantize_weight_strict_bitnet",
    "ternary_stats",
    "entropy_from_ternary_ratios",
    "symmetric_zero_ratio_for_entropy",
    "TargetBitsController",
    "SparseBranchController",
]
