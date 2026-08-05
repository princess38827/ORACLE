"""
Image and text preprocessing stubs for the OvalHumanoidVLA model.

These classes provide drop-in placeholders that implement the minimal
interface expected by the model.  Replace them with real integrations
(e.g. ``torchvision.transforms``, HuggingFace ``AutoTokenizer``) without
changing the model code.

ImagePreprocessor
-----------------
* Resizes images to ``(image_size, image_size)`` using bilinear interpolation.
* Normalises pixel values with configurable ``mean`` / ``std``.

TextTokenizer
-------------
* Maps whitespace-split words to integer IDs via a fixed vocabulary.
* Returns padded ``token_ids`` and ``attention_mask`` tensors.
* Swap in ``AutoTokenizer.from_pretrained(...)`` for production use.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


class ImagePreprocessor:
    """Minimal image pre-processing stub.

    Args:
        image_size: Target spatial resolution (both H and W).
        mean: Per-channel normalisation mean.  Defaults to ImageNet values.
        std:  Per-channel normalisation std.   Defaults to ImageNet values.
    """

    _DEFAULT_MEAN = (0.485, 0.456, 0.406)
    _DEFAULT_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: int = 224,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean or self._DEFAULT_MEAN).view(1, 3, 1, 1)
        self.std  = torch.tensor(std  or self._DEFAULT_STD ).view(1, 3, 1, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Pre-process a batch of images.

        Args:
            images: ``(B, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            ``(B, C, image_size, image_size)`` normalised float tensor.
        """
        if images.ndim != 4:
            raise ValueError(
                f"Expected 4-D image tensor (B, C, H, W), got shape {tuple(images.shape)}."
            )
        # Resize if needed
        h, w = images.shape[-2], images.shape[-1]
        if h != self.image_size or w != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        # Normalise
        mean = self.mean.to(images.device)
        std  = self.std.to(images.device)
        return (images - mean) / std


class TextTokenizer:
    """Minimal whitespace tokeniser stub.

    Builds a vocabulary from a list of words (or uses a provided mapping)
    and converts text strings to padded integer ID tensors.

    In production, replace this with ``AutoTokenizer.from_pretrained(...)``
    and keep the ``__call__`` signature identical.

    Args:
        vocab_size:   Maximum vocabulary size (entries beyond this are mapped
                      to the ``<unk>`` token).
        max_length:   Maximum token sequence length (longer sequences are
                      truncated; shorter ones are right-padded with 0).
        vocab:        Optional pre-built ``{word: id}`` mapping.  If omitted,
                      the tokeniser starts empty and learns from ``encode``
                      calls (useful for unit tests).
    """

    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3

    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 32,
        vocab: dict[str, int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        # Reserve ids 0-3 for special tokens
        self._word2id: dict[str, int] = vocab if vocab is not None else {}
        self._next_id = max(self._word2id.values(), default=3) + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_add(self, word: str) -> int:
        if word in self._word2id:
            return self._word2id[word]
        if self._next_id >= self.vocab_size:
            return self.UNK_ID
        self._word2id[word] = self._next_id
        self._next_id += 1
        return self._word2id[word]

    def encode(self, text: str) -> list[int]:
        """Tokenise a single string to a list of integer IDs."""
        ids = [self._get_or_add(w) for w in text.split()]
        ids = ids[: self.max_length - 2]   # leave room for BOS/EOS
        return [self.BOS_ID] + ids + [self.EOS_ID]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, texts: str | Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise one or more instruction strings.

        Args:
            texts: A single string or a list of strings.

        Returns:
            ``token_ids``:   ``(B, max_length)`` long tensor.
            ``attn_mask``:   ``(B, max_length)`` float tensor (1=real, 0=pad).
        """
        if isinstance(texts, str):
            texts = [texts]

        all_ids: list[list[int]] = [self.encode(t) for t in texts]

        # Pad to max_length
        token_ids = torch.full(
            (len(all_ids), self.max_length), self.PAD_ID, dtype=torch.long
        )
        attn_mask = torch.zeros(len(all_ids), self.max_length, dtype=torch.float)

        for i, ids in enumerate(all_ids):
            L = min(len(ids), self.max_length)
            token_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            attn_mask[i, :L] = 1.0

        return token_ids, attn_mask
