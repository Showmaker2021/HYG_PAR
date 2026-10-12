import pytest
import torch
from tensordict import TensorDict
from parco.models.agent_handlers import FairnessMultiRoundAgentHandler, get_agent_handler


def test_fmcr_instantiation():
    handler = get_agent_handler("fmcr", fairness_weight=0.5, max_rounds=2)
    assert isinstance(handler, FairnessMultiRoundAgentHandler)
    assert handler.fairness_weight == 0.5
    assert handler.max_rounds == 2


def test_fmcr_second_best_reproposal():
    """Verify that losers in Round 1 claim feasible second-best alternatives in Round 2."""
    handler = FairnessMultiRoundAgentHandler(fairness_weight=0.5, max_rounds=2)

    # Batch of 1, 3 agents
    # Actions: agent 0 and agent 1 BOTH choose node 5! (Collision)
    # Agent 2 chooses node 8 (No collision)
    actions = torch.tensor([[5, 5, 8]])

    # Probs over 10 nodes [B=1, M=3, N=10]
    # Agent 0: node 5 has prob 0.8, node 3 has prob 0.1
    # Agent 1: node 5 has prob 0.6, node 4 has prob 0.35 (second best!)
    # Agent 2: node 8 has prob 0.9
    probs = torch.zeros(1, 3, 10)
    probs[0, 0, 5] = 0.8
    probs[0, 0, 3] = 0.1

    probs[0, 1, 5] = 0.6
    probs[0, 1, 4] = 0.35  # Agent 1's alternative!

    probs[0, 2, 8] = 0.9

    td = TensorDict({
        "action_mask": torch.ones(1, 3, 10, dtype=torch.bool),
    }, batch_size=[1])

    final_actions, handling_mask, halting_ratio = handler(
        actions=actions,
        replacement_value=-1,
        td=td,
        probs=probs,
    )

    # Agent 0 had higher prob (0.8 vs 0.6) -> wins node 5
    assert final_actions[0, 0].item() == 5
    # Agent 1 lost node 5, but in Round 2 claimed node 4 (its second best)!
    assert final_actions[0, 1].item() == 4
    # Agent 2 had node 8 -> kept node 8
    assert final_actions[0, 2].item() == 8
    # Nobody had to halt!
    assert halting_ratio == 0.0
    assert not handling_mask.any()


def test_fmcr_fairness_counter_accumulation():
    """Verify that agents that lose are boosted in priority in subsequent steps."""
    handler = FairnessMultiRoundAgentHandler(fairness_weight=1.0, max_rounds=1)  # 1 round to force a halt

    # Step 1: Agent 0 and Agent 1 collide on node 2
    # Agent 0 has prob 0.7, Agent 1 has prob 0.6
    actions1 = torch.tensor([[2, 2]])
    probs1 = torch.zeros(1, 2, 5)
    probs1[0, 0, 2] = 0.7
    probs1[0, 1, 2] = 0.6

    final_act1, mask1, _ = handler(actions1, replacement_value=-1, probs=probs1)
    # Agent 0 wins, Agent 1 halts
    assert final_act1[0, 0].item() == 2
    assert final_act1[0, 1].item() == -1
    # Agent 1 should have fairness counter = 1.0, Agent 0 should have counter = 0.0
    assert handler.fairness_counter[0, 1].item() == 1.0
    assert handler.fairness_counter[0, 0].item() == 0.0

    # Step 2: Again collide on node 3
    # Agent 0 has prob 0.7, Agent 1 still has prob 0.6
    # But Agent 1 has priority = 0.6 + 1.0*1.0 = 1.6 > 0.7!
    # Therefore, Agent 1 MUST WIN in Step 2 due to fairness!
    actions2 = torch.tensor([[3, 3]])
    probs2 = torch.zeros(1, 2, 5)
    probs2[0, 0, 3] = 0.7
    probs2[0, 1, 3] = 0.6

    final_act2, mask2, _ = handler(actions2, replacement_value=-1, probs=probs2)
    assert final_act2[0, 1].item() == 3, "Agent 1 must win in step 2 due to fairness accumulation"
    assert final_act2[0, 0].item() == -1, "Agent 0 must yield in step 2"
