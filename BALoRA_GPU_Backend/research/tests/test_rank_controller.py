import pytest
import torch
from models.rank_controller import RankController, RankControllerConfig


def test_rank_bank_and_attribution_sensitivity():
    rc = RankController(["a", "b"], RankControllerConfig([2,4,8], learned_prior_weight=0.0))
    low = rc.select_ranks({"a":0.0,"b":0.0})
    assert set(low.values()) == {2}
    high = rc.select_ranks({"a":1.0,"b":0.0})
    assert high["a"] == 8 and high["b"] == 2
    assert all(v in {2,4,8} for v in high.values())


def test_rank_rejects_unknown_and_nonfinite():
    rc = RankController(["a"], RankControllerConfig([2,4]))
    with pytest.raises(ValueError, match="Unknown"):
        rc.select_ranks({"nope": 0.2})
    with pytest.raises(ValueError, match="finite"):
        rc.select_ranks({"a": float("nan")})


def test_rank_controller_cpu_cuda_placement():
    rc = RankController(["a"], RankControllerConfig([2,4])).cpu()
    ranks = rc.select_ranks({"a":0.5})
    assert ranks["a"] in {2,4}
    if torch.cuda.is_available():
        rc.cuda()
        assert rc.select_ranks({"a":1.0})["a"] in {2,4}
