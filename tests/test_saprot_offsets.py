"""Cross-family targets and per-position offsets.

SaProt's 20 values at a position are log-sums over its 21 structure states; ESM2's are raw head outputs. Two models of
different families therefore sit on different (and position-dependent) offsets. These tests pin which objective cares:
the score objective runs the logits through log-softmax and cannot see a constant per position, the direct-logit
objective's Huber term compares raw logits and can. This is a property of `mpd.objectives`, tested here on random numbers,
not a study of what it does to a real SaProt/ESM2 pair (docs/SAPROT.md says so).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mpd.objectives import direct_loss, readout_delta, readout_dual, score_loss

OFFSETS = torch.tensor([3.0, -2.0, 0.5, 10.0, -7.5, 1.0])[:, None]          # one constant per position


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    student, teacher = 2 * torch.randn(6, 20, generator=g), 2 * torch.randn(6, 20, generator=g)
    wt = torch.randint(0, 20, (6,), generator=g)
    row = torch.arange(6).repeat(3)                                         # three observed variants per position
    mt = torch.randint(0, 20, (18,), generator=g)
    return student, teacher, wt, row, mt


def test_a_per_position_offset_of_the_teacher_does_not_change_the_score_loss(batch):
    student, teacher, wt, row, mt = batch
    plain, _, gs, gt = score_loss(student, teacher, wt, row, mt)
    moved, _, gs2, gt2 = score_loss(student, teacher + OFFSETS, wt, row, mt)
    assert plain > 0.05 and torch.allclose(gt, gt2, atol=1e-5)               # the teacher's scalar score is the same ...
    assert torch.allclose(plain, moved, atol=1e-6) and torch.equal(gs, gs2)  # ... so the loss is, and the student is untouched


def test_a_per_position_offset_of_the_teacher_changes_the_direct_loss(batch):
    student, teacher, _, _, _ = batch
    plain, parts = direct_loss(student, teacher, 1.0)
    moved, moved_parts = direct_loss(student, teacher + OFFSETS, 1.0)
    assert torch.allclose(parts["kl"], moved_parts["kl"], atol=1e-6)          # the KL term is a softmax: offset-free
    assert abs(float(moved_parts["huber"]) - float(parts["huber"])) > 1.0       # the Huber term compares raw values
    assert abs(float(moved) - float(plain)) > 1.0


def test_the_zero_shot_readouts_do_not_see_a_per_position_offset_either(batch):
    scores, _, wt, row, mt = batch
    plain, moved = scores[row].numpy(), (scores + OFFSETS)[row].numpy()      # the scored vector at each variant's position
    wt_index, mt_index = wt[row].numpy(), mt.numpy()
    assert np.allclose(readout_delta(plain, wt_index, mt_index), readout_delta(moved, wt_index, mt_index), atol=1e-5)
    assert np.allclose(readout_dual(plain, wt_index, mt_index), readout_dual(moved, wt_index, mt_index), atol=1e-5)
