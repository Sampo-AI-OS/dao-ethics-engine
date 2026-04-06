from main import AGENT_VALUES, compute_gini, evaluate_proposal


def test_compute_gini_is_zero_for_equal_distribution():
    assert compute_gini([0.25, 0.25, 0.25, 0.25]) == 0.0


def test_compute_gini_rejects_concentrated_distribution():
    concentrated_gini = compute_gini([0.9, 0.05, 0.03, 0.02])
    assert concentrated_gini > AGENT_VALUES["required_thresholds"]["max_gini_concentration"]


def test_evaluate_proposal_approves_broadly_shared_well_supported_case():
    result = evaluate_proposal(
        proposal_name="Public Service AI Oversight",
        description="Evidence-backed proposal for broad public oversight.",
        beneficiary_distribution=[0.3, 0.25, 0.25, 0.2],
        evidence_claims=[
            {
                "claim": "Pilot improved case handling consistency",
                "sample_size": 180,
                "confidence": 0.91,
                "source": "public pilot",
                "peer_reviewed": False,
            },
            {
                "claim": "Audit process reduced manual backlog",
                "sample_size": 120,
                "confidence": 0.88,
                "source": "operations review",
                "peer_reviewed": False,
            },
        ],
    )
    assert result["recommendation"] in {"approve", "review"}
    assert result["public_benefit_score"] >= 0.5
    assert result["gini_coefficient"] <= 0.4


def test_evaluate_proposal_rejects_elite_capture_without_evidence():
    result = evaluate_proposal(
        proposal_name="Selective Incentive Program",
        description="Obvious win for a small strategic group.",
        beneficiary_distribution=[0.95, 0.05],
        evidence_claims=[],
    )
    assert result["recommendation"] == "reject"
    assert result["reasoning"]["rejection_triggers"]
