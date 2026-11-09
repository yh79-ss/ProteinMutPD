"""Model families. Each family module exposes a class with `encode`, `logits`, `trainable` and `save_adapter`."""

from __future__ import annotations

from .. import registry


def load(key: str, device, *, lora=None, adapter=None, dtype=None):
    """Load model `key` as a masked-position AA20 logit model (frozen, to train with `lora`, or with `adapter`)."""
    family = registry.get(key).family
    if family == "esm2":
        from .esm2 import Esm2
        return Esm2(key, device, dtype, lora=lora, adapter=adapter)
    if family == "saprot":
        from .saprot import Saprot
        return Saprot(key, device, dtype, lora=lora, adapter=adapter)
    raise NotImplementedError(f"no implementation for model family {family!r} yet")
