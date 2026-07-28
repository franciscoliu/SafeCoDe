"""SafeCoDe -- Safety-aware Contrastive Decoding for multimodal LLMs.

Reference implementation for "Steering Multimodal Large Language Models
Decoding for Context-Aware Safety" (arXiv:2509.19212).

Nothing heavy is imported here on purpose: ``import safecode`` must stay cheap
and must never require torch, transformers or an API key. Import the
submodules you need directly, e.g.::

    from safecode.utils import contrastive_decode_multistep_with_modulation
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
