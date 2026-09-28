"""Member 3 synthetic execution check; never reads data or updates weights.

Run from the project root: python -B -m src.models.check_bilstm_attn
"""

import torch
from torch import nn

from src.models.bilstm_attn import BiLSTMAttention


def main() -> None:
    torch.manual_seed(42)
    # Match src.data.encode: integer IDs, unknown ID 1, right-padding ID 0.
    input_ids = torch.tensor(
        [[2, 3, 4, 0, 0], [5, 1, 6, 7, 8], [9, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    model = BiLSTMAttention(vocab_size=10, n_classes=5)
    logits, attention = model(input_ids)
    real_words = input_ids.ne(0)

    assert logits.shape == (3, 5)
    assert attention.shape == input_ids.shape
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(attention[~real_words]).item() == 0
    sums = (attention * real_words).sum(dim=1)
    torch.testing.assert_close(sums, torch.ones(3), atol=1e-6, rtol=0)
    print(f"PASS: logits shape = {list(logits.shape)}")
    print(f"PASS: attention shape = {list(attention.shape)}")
    print("PASS: padded positions have exactly zero attention")
    print(f"PASS: real-word attention sums = {sums.detach().tolist()}")

    # Compute gradients once, without an optimizer or any weight updates.
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 2, 4]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient: {name}"
        torch.testing.assert_close(parameter.detach(), before[name], atol=0, rtol=0)
    assert torch.count_nonzero(model.embedding.weight.grad[0]).item() == 0
    print("PASS: backward pass; all gradients finite; weights unchanged")

    # Extra padding must not change a complaint's prediction.
    model.eval()
    with torch.no_grad():
        short_logits, short_attention = model(input_ids[:1, :3])
        padded_logits, padded_attention = model(input_ids[:1])
        torch.testing.assert_close(short_logits, padded_logits)
        torch.testing.assert_close(short_attention, padded_attention[:, :3])
    print("PASS: extra right-padding does not change predictions or word attention")

    # An empty complaint has no real words, so its attention sums to zero.
    model.zero_grad(set_to_none=True)
    empty_logits, empty_attention = model(torch.zeros((1, 5), dtype=torch.long))
    assert torch.isfinite(empty_logits).all()
    assert torch.count_nonzero(empty_attention).item() == 0
    empty_logits.sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    print("PASS: all-padding input has zero attention and a finite backward pass")


if __name__ == "__main__":
    main()
