import torch

from chimera import Chimera21


def test_optimizer_step_changes_parameter() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = Chimera21([p], lr=0.1)
    before = p.detach().clone()

    p.grad = torch.tensor([0.25, -0.5])
    opt.step()

    assert not torch.allclose(p.detach(), before)
    assert not hasattr(opt, "current_alpha")
