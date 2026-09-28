"""
test_transformer.py — sanity tests for Member 4's from-scratch Transformer.

Synthetic data only: never touches data/*.csv, so it cannot leak the test set.

    python scripts/test_transformer.py

Runs every check on CPU, then again on CUDA if a GPU is available.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import torch
import torch.nn as nn

# make `src` importable when run as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VOCAB, N_CLASSES, BATCH, SEQ = 1000, 5, 4, 256
CONFIG = dict(n_classes=N_CLASSES, d_model=256, n_heads=4, n_layers=4,
              d_ff=1024, dropout=0.1, max_len=256)

results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run a check, record pass/fail with its detail line."""
    def wrap(fn):
        def run(*args):
            try:
                detail = fn(*args) or ""
                results.append((name, True, detail))
            except Exception as e:                      # noqa: BLE001
                results.append((name, False, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        return run
    return wrap


def synthetic_batch(device: torch.device, seed: int = 0):
    """Post-padded ids like src.data.encode: real tokens in 2..VOCAB-1, pad = 0."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([SEQ, 180, 60, 10])
    ids = torch.randint(2, VOCAB, (BATCH, SEQ), generator=g)
    pad = torch.arange(SEQ)[None, :] >= lengths[:, None]
    ids[pad] = 0
    labels = torch.randint(0, N_CLASSES, (BATCH,), generator=g)
    return ids.to(device), (~pad).to(device), labels.to(device), lengths


def all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


# --------------------------------------------------------------------------
# device-independent checks
# --------------------------------------------------------------------------

@check("1. model imports")
def test_import():
    import src.models.transformer as m
    for cls in ("PositionalEncoding", "MultiHeadSelfAttention", "FeedForward",
                "TransformerEncoderBlock", "TransformerClassifier"):
        assert hasattr(m, cls), f"missing {cls}"
    return "all 5 classes present"


@check("2. model instantiates (4 layers, 4 heads, d_model 256, d_ff 1024)")
def test_instantiate():
    from src.models.transformer import TransformerClassifier
    model = TransformerClassifier(vocab_size=VOCAB, **CONFIG)
    assert len(model.layers) == 4
    blk = model.layers[0]
    assert blk.attn.n_heads == 4 and blk.attn.d_k == 64
    assert blk.ff.fc1.out_features == 1024
    assert model.classifier.out_features == N_CLASSES
    return "config " + str({k: model.config[k] for k in
                           ("n_layers", "n_heads", "d_model", "d_ff", "n_classes")})


@check("13. parameter count")
def test_params():
    from src.models.transformer import TransformerClassifier
    lines = []
    for vocab in (VOCAB, 30000):            # test vocab, and build_vocab's max size
        model = TransformerClassifier(vocab_size=vocab, **CONFIG)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb = model.embedding.weight.numel()
        lines.append(f"vocab {vocab:>6,}: total {total:,} | trainable {trainable:,} | "
                     f"embedding {emb:,} | rest {total - emb:,}")
    return "\n      ".join(lines)


# --------------------------------------------------------------------------
# per-device checks
# --------------------------------------------------------------------------

def device_suite(device: torch.device, label: str):
    from src.models.transformer import TransformerClassifier

    torch.manual_seed(42)
    model = TransformerClassifier(vocab_size=VOCAB, **CONFIG).to(device)
    ids, mask, labels, lengths = synthetic_batch(device)

    @check(f"3. [{label}] input shape [batch, 256]")
    def t_input():
        assert ids.shape == (BATCH, 256), ids.shape
        assert ids.dtype == torch.long
        return f"{tuple(ids.shape)} {ids.dtype}, lengths {lengths.tolist()}"

    @check(f"6. [{label}] forward pass")
    def t_forward():
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert all_finite(out)
        return f"logits range [{out.min().item():.3f}, {out.max().item():.3f}]"

    @check(f"4. [{label}] output shape [batch, 5]")
    def t_output():
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert out.shape == (BATCH, N_CLASSES), out.shape
        return str(tuple(out.shape))

    @check(f"5. [{label}] padding mask works")
    def t_mask():
        model.eval()
        with torch.no_grad():
            # (a) explicit mask == mask derived from pad id 0
            a, b = model(ids, mask), model(ids)
            assert torch.allclose(a, b, atol=1e-5)
            # (b) no attention weight lands on padded keys
            _, atts = model(ids, mask, return_attention=True)
            key_pad = (~mask)[:, None, None, :]
            leak = max(w.masked_select(key_pad.expand_as(w)).abs().max().item()
                       for w in atts)
            assert leak == 0.0, leak
            # (c) extra padding does not change the prediction: the 60- and
            #     10-token rows padded to 64 vs to 256 give the same logits
            short = model(ids[2:, :64], mask[2:, :64])
            long_ = model(ids[2:], mask[2:])
            diff = (short - long_).abs().max().item()
            assert diff < 1e-4, diff
            # (d) changing ids under the mask does not change the output
            noisy = ids.clone()
            noisy[~mask] = torch.randint(2, VOCAB, (int((~mask).sum()),), device=device)
            diff2 = (model(noisy, mask) - a).abs().max().item()
            assert diff2 < 1e-4, diff2
        return (f"explicit==derived; attention on pads {leak:.1f}; "
                f"64 vs 256 padding diff {diff:.1e}; tokens under mask diff {diff2:.1e}")

    loss_fn = nn.CrossEntropyLoss()
    state: dict = {}

    @check(f"7. [{label}] CrossEntropyLoss")
    def t_loss():
        model.train()
        loss = loss_fn(model(ids, mask), labels)
        assert loss.dim() == 0 and all_finite(loss)
        # untrained 5-class model should sit near ln(5) = 1.609
        assert 0.5 < loss.item() < 5.0, loss.item()
        # class-weighted variant, as the training script will use
        w = torch.tensor([1.0, 0.8, 1.2, 1.0, 1.5], device=device)
        wloss = nn.CrossEntropyLoss(weight=w)(model(ids, mask), labels)
        assert all_finite(wloss)
        state["loss"] = loss
        return f"loss {loss.item():.4f} (ln 5 = {math.log(5):.4f}); weighted {wloss.item():.4f}"

    @check(f"8. [{label}] backward pass")
    def t_backward():
        model.zero_grad(set_to_none=True)
        state["loss"].backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        bad = [n for n, p in model.named_parameters() if not all_finite(p.grad)]
        assert not bad, bad
        gnorm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())).item()
        assert gnorm > 0
        pad_grad = model.embedding.weight.grad[0].abs().sum().item()
        assert pad_grad == 0.0, pad_grad
        n = sum(1 for _ in model.parameters())
        return f"{n}/{n} tensors have finite grads, norm {gnorm:.3f}, <pad> row grad 0"

    @check(f"9. [{label}] optimizer step")
    def t_step():
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        changed = sum(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        assert changed > 0
        # a few more steps on the fixed batch should lower the loss
        model.eval()
        with torch.no_grad():
            l0 = loss_fn(model(ids, mask), labels).item()
        model.train()
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            loss_fn(model(ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            l1 = loss_fn(model(ids, mask), labels).item()
        assert l1 < l0, (l0, l1)
        assert model.embedding.weight[0].abs().sum().item() == 0.0
        return (f"{changed} tensors updated; 10 steps: loss {l0:.4f} -> {l1:.4f}; "
                "<pad> embedding stays zero")

    @check(f"10. [{label}] no NaN / inf")
    def t_finite():
        model.eval()
        with torch.no_grad():
            out, atts = model(ids, mask, return_attention=True)
            probs = torch.softmax(out, -1)
        assert all_finite(out) and all_finite(probs)
        assert all(all_finite(w) for w in atts)
        assert all(all_finite(p) for p in model.parameters())
        assert torch.allclose(probs.sum(-1), torch.ones(BATCH, device=device), atol=1e-5)
        extra = ""
        if device.type == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                half = model(ids, mask)
            assert all_finite(half), "fp16 autocast produced NaN/inf"
            extra = "; fp16 autocast finite"
        return "logits, probabilities, attention weights and parameters all finite" + extra

    for t in (t_input, t_forward, t_output, t_mask, t_loss, t_backward, t_step, t_finite):
        t()


def main() -> int:
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}\n")

    test_import()
    test_instantiate()

    @check("11. model works on CPU")
    def t_cpu():
        n_before = sum(1 for r in results if not r[1])
        device_suite(torch.device("cpu"), "cpu")
        fails = sum(1 for r in results if not r[1]) - n_before
        assert fails == 0, f"{fails} CPU check(s) failed"
        return "all CPU checks passed"
    t_cpu()

    @check("12. model works on CUDA")
    def t_cuda():
        if not torch.cuda.is_available():
            return "SKIPPED: CUDA not available on this machine"
        n_before = sum(1 for r in results if not r[1])
        device_suite(torch.device("cuda"), "cuda")
        fails = sum(1 for r in results if not r[1]) - n_before
        assert fails == 0, f"{fails} CUDA check(s) failed"
        return f"all CUDA checks passed on {torch.cuda.get_device_name(0)}"
    t_cuda()

    test_params()

    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        status = "SKIP" if detail.startswith("SKIPPED") else ("PASS" if ok else "FAIL")
        print(f"{status}  {name:<{width}}  {detail}")
    n_fail = sum(1 for r in results if not r[1])
    n_skip = sum(1 for r in results if r[2].startswith("SKIPPED"))
    print(f"\n{len(results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
