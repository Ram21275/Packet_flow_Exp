import torch

from packets_exp.audit import (
    DirectCostScorer,
    _diagnostic_signals,
    masks_to_actions,
    omission_actions,
    policy_metrics,
    teacher_stability,
)


def test_omission_actions_round_trip_for_one_and_two_drops():
    for drops, expected in [(1, 5), (2, 10)]:
        combinations, masks = omission_actions(5, drops)
        assert len(combinations) == expected
        assert torch.all(masks.sum(dim=1) == 5 - drops)
        assert torch.equal(masks_to_actions(masks, masks), torch.arange(expected))


def test_teacher_stability_detects_stable_and_unstable_labels():
    draws = torch.tensor([
        [[0., 1., 2.], [0., 1., 2.]],
        [[0., 1., 2.], [1., 0., 2.]],
        [[0., 1., 2.], [2., 1., 0.]],
    ])
    result = teacher_stability(draws)
    assert result["modal_agreement"][0] == 1
    assert torch.isclose(result["modal_agreement"][1], torch.tensor(1 / 3))
    assert result["normalized_entropy"][0] == 0
    assert torch.isclose(result["normalized_entropy"][1], torch.tensor(1.0))


def test_policy_metrics_measure_oracle_agreement_and_regret():
    costs = torch.tensor([[1., 2., 3.], [3., 1., 2.]])
    oracle = torch.tensor([0, 1])
    wrong = torch.tensor([2, 2])
    assert policy_metrics(costs, oracle)["mean_regret"] == 0
    result = policy_metrics(costs, wrong)
    assert result["oracle_agreement"] == 0
    assert result["mean_regret"] == 1.5


def test_direct_cost_scorer_outputs_one_cost_per_action():
    model = DirectCostScorer(dim=8, actions=5, width=16, context=4)
    result = model(torch.randn(3, 5, 8), torch.randn(3, 25, 8))
    assert result.shape == (3, 5)
    result.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_diagnostic_signals_flag_a_bad_flow_formulation():
    base = {
        "oracle": {"mean_cost": 1.0, "mean_regret": 0.0},
        "energy": {"mean_cost": 1.2, "mean_regret": .2},
        "flow": {"mean_cost": 1.5, "mean_regret": .5},
    }
    stability = {"mean_modal_agreement": .5}
    integration = [
        {"steps": 1, "mean_regret": .2},
        {"steps": 1, "mean_regret": .45},
        {"steps": 2, "mean_regret": .6},
        {"steps": 2, "mean_regret": .65},
    ]
    direct = {"mean_regret": .1}
    signals = _diagnostic_signals(base, stability, integration, direct)
    assert signals["teacher_unstable"]
    assert signals["selector_worse_than_energy"]
    assert signals["direct_baseline_beats_selector"]
    assert signals["integration_sensitive"]
    assert signals["selector_noise_sensitive"]
    assert signals["oracle_has_energy_headroom"]
    assert signals["formulation_concern_count"] == 6
