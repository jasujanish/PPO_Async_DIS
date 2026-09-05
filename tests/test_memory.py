import torch

from ppo_async.training.memory import checkpoint_chunks


def test_chunk_checkpoint_preserves_masked_logprob_gradients_without_saved_softmax():
    def calculate(logits, targets, mask):
        selected = logits.masked_fill(~mask, float('-inf'))
        return selected.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

    torch.manual_seed(1)
    original = torch.randn(12, 32, requires_grad=True)
    targets = torch.arange(12)
    mask = torch.rand(12, 32) > 0.4
    mask[torch.arange(12), targets] = True
    expected = calculate(original, targets, mask)
    expected.sum().backward()
    actual_input = original.detach().clone().requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: (saved.append(tensor), tensor)[1], lambda t: t):
        actual = checkpoint_chunks(calculate)(actual_input, targets, mask)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(actual_input.grad, original.grad)
    # Inputs are retained by reference; no extra full-vocabulary floating tensor.
    assert all(t.numel() == 0 or not t.is_floating_point() or t.data_ptr() == actual_input.data_ptr() for t in saved)
