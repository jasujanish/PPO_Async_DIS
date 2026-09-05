"""Recompute each vocabulary-softmax chunk instead of retaining all chunks."""

from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_chunks(function):
    @wraps(function)
    def wrapped(logits, *args, **kwargs):
        if torch.is_grad_enabled() and logits.requires_grad:
            return checkpoint(function, logits, *args, use_reentrant=False,
                              preserve_rng_state=False, **kwargs)
        return function(logits, *args, **kwargs)
    return wrapped


def initialize(args):
    # SLIME's existing chunks otherwise retain a full [tokens, vocabulary]
    # softmax for backward. Its supported worker-init hook installs this once
    # per process; generation and inference-only forwards are unaffected.
    from slime.utils import ppo_utils

    function = ppo_utils._calculate_log_probs_and_entropy_chunk
    if not getattr(function, '_ppo_checkpointed', False):
        wrapped = checkpoint_chunks(function)
        wrapped._ppo_checkpointed = True
        ppo_utils._calculate_log_probs_and_entropy_chunk = wrapped
