"""Checkpoint save/resume roundtrip: a resumed model+optimizer must continue the
exact same trajectory as an uninterrupted run (the property that makes long runs
recoverable). Tests the mechanism on a tiny model, no corpus build."""
import torch

from terrace.model import ModelArgs, TerraceLM


def _tiny():
    a = ModelArgs(vocab=64, dim=32, n_layers=4, n_heads=4, seq_len=16,
                  n_experts=16, n_groups=4, top_k=4, top_groups=2,
                  d_expert=16, d_shared=24, n_shared=1, expert_backend="grouped")
    return a


def _steps(model, opt, gen, xs, n):
    model.train()
    for i in range(n):
        x = torch.randint(0, 64, (4, 16), generator=gen)
        y = torch.randint(0, 64, (4, 16), generator=gen)
        loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss.item()


def test_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(0)
    a = _tiny()

    # uninterrupted 10-step reference
    torch.manual_seed(1); m0 = TerraceLM(a); o0 = torch.optim.AdamW(m0.parameters(), lr=1e-3)
    g0 = torch.Generator().manual_seed(7)
    _steps(m0, o0, g0, None, 10)
    ref = torch.nn.utils.parameters_to_vector(m0.parameters()).detach()

    # split run: 6 steps, checkpoint, resume, 4 more
    torch.manual_seed(1); m1 = TerraceLM(a); o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    g1 = torch.Generator().manual_seed(7)
    _steps(m1, o1, g1, None, 6)
    ck = tmp_path / "c.ckpt"
    torch.save({"model": m1.state_dict(), "opt": o1.state_dict(), "gen": g1.get_state()}, ck)

    torch.manual_seed(1); m2 = TerraceLM(a); o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    saved = torch.load(ck)
    m2.load_state_dict(saved["model"]); o2.load_state_dict(saved["opt"])
    g2 = torch.Generator(); g2.set_state(saved["gen"])
    _steps(m2, o2, g2, None, 4)
    got = torch.nn.utils.parameters_to_vector(m2.parameters()).detach()

    assert torch.allclose(got, ref, atol=1e-5), \
        f"resumed run diverged: max|Δ|={ (got-ref).abs().max():.2e}"
