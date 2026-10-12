import abc

from typing import Callable, List, Union, Tuple

import torch

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AgentHandler(abc.ABC):
    """Base class for agent handlers. Handles conflicts between agents.
    By default, one occurrence is always kept (i.e. one agent among agents
    that selected the same action is selected).

    Args:
        mask_all: If True, the all occurrences of the same value will be masked, i.e. no agent will select it.
        exclude_values: If provided, the values in the actions that are in this list will not be masked.
        return_none_mask: If True, the mask will be None. This may be useful for loss computation.
    """

    def __init__(
        self,
        mask_all: bool = False,
        exclude_values: Union[torch.Tensor, int, List] = None,
        return_none_mask: bool = False,
    ):
        super(AgentHandler, self).__init__()
        self.mask_all = mask_all
        self.exclude_values = exclude_values
        self.return_none_mask = return_none_mask

    @abc.abstractmethod
    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Preprocesses actions such that first action in order to appear with an index will be selected"""
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(
        self,
        actions: torch.Tensor,
        replacement_value: Union[torch.Tensor, int] = -1,
        td=None,
        exclude_values: Union[torch.Tensor, int, List] = None,
        probs: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        exclude_values = self.exclude_values if exclude_values is None else exclude_values
        if isinstance(exclude_values, int):
            exclude_values = [exclude_values]
        if not isinstance(exclude_values, torch.Tensor) and exclude_values is not None:
            exclude_values = torch.tensor(exclude_values, device=actions.device)

        # First reordering of actions based on the type of handler (highest probability, closest, etc.)
        sorted_actions1, indices1 = self._preprocess_actions(
            actions, td, probs=probs, **kwargs
        )

        # Second reordering of actions based on the index of selected nodes, for masking non-first occurrences
        sorted_actions2, indices2 = sorted_actions1.sort(dim=1)

        # Create a mask for non-first occurrences on the sorted actions
        mask_sorted = torch.zeros_like(actions, dtype=torch.bool)
        mask_sorted[:, 1:] = sorted_actions2[:, 1:] == sorted_actions2[:, :-1]

        # Mask first occurrences on the sorted actions if needed
        if self.mask_all:
            mask_sorted[:, :-1] |= sorted_actions2[:, :-1] == sorted_actions2[:, 1:]

        # Recover the mask_sorted based on the second reordering of actions
        mask_sorted = mask_sorted.gather(1, indices2.argsort(dim=1))

        # Recover the mask_sorted based on the first reordering of actions
        mask = mask_sorted.gather(1, indices1.argsort(dim=1))

        # If exclude_values is provided, we set their mask to False so that they are not replaced
        if exclude_values is not None:
            exclude_values = exclude_values.to(actions.device)
            mask = mask & ~torch.isin(actions, exclude_values)

        # Replace values in the original actions using the mask
        if isinstance(replacement_value, int):
            actions[mask] = replacement_value
        else:
            actions[mask] = replacement_value[mask]

        # Calculate num of conflicts (sum of true values in mask / total)
        halting_ratio = mask.sum().float() / (mask.numel())
        return actions, mask if not self.return_none_mask else None, halting_ratio


class FirstPrecedenceAgentHandler(AgentHandler):
    """First agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(self, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        indices = torch.arange(actions.size(1), device=actions.device).repeat(
            actions.size(0), 1
        )
        return actions, indices


class RandomAgentHandler(AgentHandler):
    """Random agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(self, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        indices_ = torch.randperm(actions.size(1), device=actions.device).repeat(
            actions.size(0), 1
        )
        shuffled_actions = torch.gather(actions, 1, indices_)
        return shuffled_actions, indices_


class ClosestAgentHandler(AgentHandler):
    """Closest agent to target node in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(
        self, actions: torch.Tensor, td, *args, **kwargs
    ) -> torch.Tensor:
        current_loc = gather_by_index(td["locs"], td["current_node"])
        target_loc = gather_by_index(td["locs"], actions)
        distances = torch.norm(current_loc - target_loc, dim=-1)
        _, indices = torch.sort(distances, dim=-1, descending=False, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class HighestProbabilityAgentHandler(AgentHandler):
    """Highest probability agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs, randomize: bool = False
    ) -> torch.Tensor:
        # sort indices by probability
        action_probs = gather_by_index(probs, actions, dim=-1)
        if randomize:
            # Add Gumbel noise to probabilities to randomize selection
            # This may be useful for exploration
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_probs)))
            action_probs += gumbel_noise
        # Sort indices by action probabilities
        _, indices = torch.sort(action_probs, dim=-1, descending=True, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class POSAgentHandler(AgentHandler):
    """Agent with **lowest** probability of selecting the POS action is put first"""

    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs=None, **kwargs
    ) -> torch.Tensor:
        # note: actually logprobs
        _, sort_pos_indices = probs.sort(
            dim=-1, descending=False
        )  # False cuz lowest POS has to be selected
        sorted_actions = gather_by_index(actions, sort_pos_indices)
        return sorted_actions, sort_pos_indices


class SmallestPathToClosure(AgentHandler):
    """Use agent that has the lowest current path"""

    def __init__(self, *args, count_depot=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_depot = count_depot

    def _preprocess_actions(self, actions: torch.Tensor, td, probs) -> torch.Tensor:
        # sort indices by probability
        current_loc = gather_by_index(td["locs"], td["current_node"])
        target_loc = gather_by_index(td["locs"], actions)
        distances_to_target = torch.norm(current_loc - target_loc, dim=-1)
        if self.count_depot:
            distance_target_depot = torch.norm(
                target_loc - td["locs"][..., 0:1, :], dim=-1
            )
        else:
            distance_target_depot = 0
        distance_target_depot = torch.norm(
            target_loc - td["locs"][..., 0:1, :], dim=-1
        )  # just singled depot
        current_traveled = td["current_length"]
        total_distance = distances_to_target + current_traveled + distance_target_depot
        _, indices = torch.sort(total_distance, dim=-1, descending=False, stable=True)
        sorted_actions = gather_by_index(actions, indices)
        return sorted_actions, indices


class LowestLateness(SmallestPathToClosure):
    def _preprocess_actions(self, actions: torch.Tensor, td, probs) -> torch.Tensor:
        # TODO
        raise NotImplementedError("Not implemented yet.")


class NoHandler(AgentHandler):
    """No handler is used, i.e. all agents can select the same action. The mask is always None."""

    def __call__(self, actions: torch.Tensor, *args, **kwargs):
        return actions, None


class FairnessMultiRoundAgentHandler(AgentHandler):
    """Fairness-Aware Multi-Round Conflict Resolver (FMCR).

    Extends PARCO's priority conflict handler with:
    1. Long-term Fairness Accumulation: Prevents agent starvation by boosting
       priority for agents that were previously forced to halt.
       priority_i = prob_i + gamma * fairness_counter_i
    2. Multi-Round Second-Best Re-proposal: In case of collisions, losing agents
       in Round 1 select their highest-probability unallocated alternative in Round 2
       instead of immediately being forced into wasteful halting.

    Args:
        fairness_weight (float): Weight gamma for fairness counter. Default 0.5.
        decay_rate (float): Decay factor alpha for winners (0.0 = full reset). Default 0.5.
        max_rounds (int): Maximum conflict resolution rounds per step. Default 2.
    """

    def __init__(
        self,
        fairness_weight: float = 0.5,
        decay_rate: float = 0.5,
        max_rounds: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fairness_weight = fairness_weight
        self.decay_rate = decay_rate
        self.max_rounds = max_rounds
        self.fairness_counter = None

    def reset(self):
        """Reset fairness counter for a new rollout/batch."""
        self.fairness_counter = None

    def _preprocess_actions(self, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Handled in custom __call__
        indices = torch.arange(actions.size(1), device=actions.device).repeat(actions.size(0), 1)
        return actions, indices

    def __call__(
        self,
        actions: torch.Tensor,
        replacement_value: Union[torch.Tensor, int] = -1,
        td=None,
        exclude_values: Union[torch.Tensor, int, List] = None,
        probs: torch.Tensor = None,
        randomize: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        exclude_values = self.exclude_values if exclude_values is None else exclude_values
        if isinstance(exclude_values, int):
            exclude_values = [exclude_values]
        if not isinstance(exclude_values, torch.Tensor) and exclude_values is not None:
            exclude_values = torch.tensor(exclude_values, device=actions.device)

        B, M = actions.shape
        device = actions.device

        # Initialize fairness counter if not yet created or batch size changed
        if (
            self.fairness_counter is None
            or self.fairness_counter.shape != actions.shape
            or self.fairness_counter.device != device
        ):
            self.fairness_counter = torch.zeros(B, M, device=device, dtype=torch.float32)

        # Base action probability / priority
        if probs is not None:
            if probs.max() <= 0:
                prob_vals = probs.exp()
            else:
                prob_vals = probs
            action_probs = gather_by_index(prob_vals, actions, dim=-1)
        else:
            action_probs = torch.ones_like(actions, dtype=torch.float32)
            prob_vals = None

        if randomize:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_probs).clamp_min(1e-9)))
            action_probs = action_probs + gumbel_noise

        # Total Priority = instantaneous confidence + long-term fairness score
        priorities = action_probs + self.fairness_weight * self.fairness_counter

        final_actions = actions.clone()
        resolved_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        claimed_nodes = [set() for _ in range(B)]

        # --- ROUND 1: Initial Proposals ---
        _, sorted_agent_idx = torch.sort(priorities, dim=-1, descending=True, stable=True)

        for b in range(B):
            for agent_idx in sorted_agent_idx[b]:
                agent = agent_idx.item()
                act = actions[b, agent].item()
                is_excludable = (exclude_values is not None) and (act in exclude_values)
                if is_excludable or (act not in claimed_nodes[b]):
                    resolved_mask[b, agent] = True
                    if not is_excludable:
                        claimed_nodes[b].add(act)

        # --- ROUND 2+: Second-Best Re-proposal for Collision Losers ---
        curr_round = 1
        while curr_round < self.max_rounds and (~resolved_mask).any():
            curr_round += 1
            if prob_vals is None or td is None:
                break

            action_mask = td.get("action_mask", None) if isinstance(td, dict) or hasattr(td, "get") else None

            for b in range(B):
                unresolved = [ag.item() for ag in sorted_agent_idx[b] if not resolved_mask[b, ag]]
                for agent in unresolved:
                    p_agent = prob_vals[b, agent].clone()
                    if action_mask is not None:
                        p_agent.masked_fill_(~action_mask[b, agent].bool(), -1e9)

                    # Mask out all actions already claimed by Round 1/earlier winners
                    for claimed in claimed_nodes[b]:
                        p_agent[claimed] = -1e9

                    # Mask out the action that was rejected in Round 1
                    p_agent[actions[b, agent]] = -1e9

                    best_alt_prob, best_alt_act = p_agent.max(dim=-1)
                    if best_alt_prob > -1e8:  # Feasible alternative found
                        alt_act = best_alt_act.item()
                        is_excludable = (exclude_values is not None) and (alt_act in exclude_values)
                        final_actions[b, agent] = alt_act
                        resolved_mask[b, agent] = True
                        if not is_excludable:
                            claimed_nodes[b].add(alt_act)

        # Halted agents are those still unresolved after max_rounds
        handling_mask = ~resolved_mask

        # Apply replacement value (e.g. current_node / depot)
        if isinstance(replacement_value, int):
            final_actions[handling_mask] = replacement_value
        else:
            final_actions[handling_mask] = replacement_value[handling_mask]

        # Update Fairness Counter
        self.fairness_counter[resolved_mask] = self.fairness_counter[resolved_mask] * self.decay_rate
        self.fairness_counter[handling_mask] = self.fairness_counter[handling_mask] + 1.0

        halting_ratio = handling_mask.sum().float() / handling_mask.numel()
        return final_actions, handling_mask if not self.return_none_mask else None, halting_ratio


AGENT_HANDLER_REGISTRY = {
    "first": FirstPrecedenceAgentHandler,
    "random": RandomAgentHandler,
    "closest": ClosestAgentHandler,
    "highprob": HighestProbabilityAgentHandler,
    "smallestpath": SmallestPathToClosure,
    "pos": POSAgentHandler,
    "fmcr": FairnessMultiRoundAgentHandler,
    "fairness": FairnessMultiRoundAgentHandler,
    "none": NoHandler,
}


def get_agent_handler(
    name: str, registry: dict = AGENT_HANDLER_REGISTRY, **config
) -> Callable:
    return registry[name](**config)

