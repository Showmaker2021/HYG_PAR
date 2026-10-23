from random import randint
from typing import Any

import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.symnco import SymNCO
from rl4co.models.zoo.symnco.losses import invariance_loss, solution_symmetricity_loss
from rl4co.utils.ops import unbatchify
from rl4co.utils.pylogger import get_pylogger
from torchrl.modules.models import MLP

from .utils import resample_batch

log = get_pylogger(__name__)


class PARCORLModule(SymNCO):
    """RL LightningModule for PARCO based on RL4COLitModule"""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: str = "symnco",
        baseline_kwargs={},
        num_augment: int = 4,
        alpha: float = 0.2,
        beta: float = 1,
        projection_head: nn.Module = None,
        use_projection_head: bool = True,
        train_min_agents: int = 5,
        train_max_agents: int = 15,
        train_min_size: int = 50,
        train_max_size: int = 100,
        val_test_num_agents: int = 10,
        allow_multi_dataloaders: bool = True,
        aux_loss_weight: float = 0.0,
        lambda_int: float = 0.0,
        lambda_col: float = 0.0,
        lambda_bal: float = 0.0,
        lambda_sp: float = 0.0,
        warmup_ratio: float = 0.1,
        lr_gate: float = 1e-3,
        **kwargs,
    ):
        # Pass no baseline to superclass since there are multiple custom baselines
        super().__init__(env, policy, **kwargs)

        self.aux_loss_weight = aux_loss_weight
        self.lambda_int = lambda_int if lambda_int > 0.0 else aux_loss_weight
        self.lambda_col = lambda_col
        self.lambda_bal = lambda_bal
        self.lambda_sp = lambda_sp
        self.warmup_ratio = warmup_ratio
        self.lr_gate = lr_gate

        self.num_augment = num_augment
        self.augment = StateAugmentation(num_augment=self.num_augment)
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss
        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            if projection_head is None:
                embed_dim = self.policy.decoder.embed_dim
                projection_head = (
                    MLP(embed_dim, embed_dim, 1, embed_dim, nn.ReLU)
                    if projection_head is None
                    else projection_head
                )
            self.projection_head = projection_head

        # Multiagent training
        self.train_min_agents = train_min_agents
        self.train_max_agents = train_max_agents
        self.train_min_size = train_min_size
        self.train_max_size = train_max_size
        self.val_test_num_agents = val_test_num_agents
        self.allow_multi_dataloaders = allow_multi_dataloaders

        # Force env to have as num_agents and num_locs as the maximum number of agents and locations
        self.env.generator.num_loc = self.train_max_size
        self.env.generator.num_agents = self.train_max_agents

    def configure_optimizers(self, parameters=None):
        if parameters is None:
            gate_params = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "gate" in name:
                    gate_params.append(param)
                else:
                    other_params.append(param)

            if len(gate_params) > 0 and self.lr_gate is not None:
                base_lr = self.optimizer_kwargs.get("lr", 1e-4)
                opt_kwargs_other = {k: v for k, v in self.optimizer_kwargs.items() if k != "lr"}
                parameters = [
                    {"params": other_params, "lr": base_lr, **opt_kwargs_other},
                    {"params": gate_params, "lr": self.lr_gate, "weight_decay": 0.0},
                ]
            else:
                parameters = self.parameters()
        return super().configure_optimizers(parameters=parameters)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # NOTE: deprecated
        num_agents = None  # done inside the sampling
        # Sample number of agents during training step
        if phase == "train":
            # Idea: we always have batches of the same size from the dataloader.
            # however, here we sample a subset of agents and locations from the batch.
            # For instance: if we have always 10 depots and 100 cities, we sample a random number of depots and cities
            # from the batch. This way, we can train on different number of agents and locations.
            num_agents = randint(self.train_min_agents, self.train_max_agents)
            num_locs = randint(self.train_min_size, self.train_max_size)
            batch = resample_batch(batch, num_agents, num_locs)
        else:
            if self.allow_multi_dataloaders:
                # Get number of agents to test on based on dataloader name
                if dataloader_idx is not None and self.dataloader_names is not None:
                    # e.g. n50_m7 take only number after "m" until _
                    num_agents = int(
                        self.dataloader_names[dataloader_idx].split("_")[-1][1:]
                    )
                else:
                    num_agents = self.val_test_num_agents

                # NOTE: trick: we subsample number of agents by setting num_agents
                # in such case, use the same number of agents for all batches
                batch["num_agents"] = torch.full(
                    (batch.shape[0],), num_agents, device=batch.device
                )

        # Reset env based on the number of agents
        td = self.env.reset(batch)

        n_aug, n_start = self.num_augment, self.num_starts
        assert n_start <= 1, "PARCO does not support multi-start"

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td,
            self.env,
            phase=phase,
            return_init_embeds=True,
        )

        # Unbatchify reward to [batch_size, n_start, n_aug].
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # Main training loss
        if phase == "train":
            # note that we do not collect the max_rewards during training
            # [batch_size, n_start, n_aug]
            ll = unbatchify(out["log_likelihood"], (n_start, n_aug))

            # Get proj_embeddings if projection head is used
            proj_embeds = self.projection_head(out["init_embeds"])
            loss_inv = invariance_loss(proj_embeds, n_aug) if n_aug > 1 else 0

            # No problem symmetricity loss since no multi-start
            loss_ps = 0

            # IMPORTANT: need to change the dimension on which to calculate the loss because of casting
            loss_ss = (
                solution_symmetricity_loss(reward[..., None], ll, dim=1)
                if n_aug > 1
                else 0
            )
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            # Auxiliary Conflict-Driven Supervision Loss (CD-HG v2.1)
            loss_aux = torch.tensor(0.0, device=reward.device)
            loss_int = torch.tensor(0.0, device=reward.device)
            loss_col = torch.tensor(0.0, device=reward.device)
            loss_bal = torch.tensor(0.0, device=reward.device)
            loss_sp = torch.tensor(0.0, device=reward.device)
            pair_auc = torch.tensor(0.5, device=reward.device)

            context_emb = getattr(self.policy.decoder, "context_embedding", None)
            comm = getattr(context_emb, "communication_layers", None) if context_emb is not None else None
            H = getattr(comm, "last_H", None) if comm is not None else None
            p = getattr(comm, "last_p", None) if comm is not None else None

            if comm is not None and H is not None:
                from parco.models.hscl_v2.losses import (
                    intent_overlap_loss,
                    realized_collision_loss,
                    anti_collapse_losses,
                    compute_pair_auc,
                )

                # Linear warmup factor (10% first epochs/steps)
                total_steps = 1000
                trainer = getattr(self, "_trainer", None)
                if trainer is not None and getattr(trainer, "estimated_stepping_batches", None):
                    total_steps = self.trainer.estimated_stepping_batches
                warmup_steps = max(1, int(total_steps * self.warmup_ratio))
                cur_step = getattr(self, "global_step", warmup_steps)
                warmup_factor = min(1.0, max(0.0, cur_step / warmup_steps))

                # Update prior beta_t (beta_t decays linearly to 0 after warmup)
                if hasattr(comm, "set_beta"):
                    decay_factor = max(0.0, 1.0 - (cur_step / warmup_steps))
                    comm.set_beta(comm.initial_beta * decay_factor)

                # 1. L_int
                if self.lambda_int > 0.0 and p is not None:
                    loss_int = intent_overlap_loss(H, p)

                # 2. L_col (computed on tentative actions BEFORE conflict handler)
                actions_init = out.get("actions_init", out.get("actions", None))
                if actions_init is not None:
                    if actions_init.dim() == 3:
                        num_steps = actions_init.shape[-1]
                        hist_H = getattr(comm, "history_H", None)
                        if hist_H is not None and len(hist_H) == num_steps:
                            step_losses = []
                            step_aucs = []
                            for s in range(num_steps):
                                act_s = actions_init[..., s]
                                H_s = hist_H[s]
                                if self.lambda_col > 0.0:
                                    step_losses.append(realized_collision_loss(H_s, act_s))
                                auc_s = compute_pair_auc(H_s, act_s)
                                if auc_s != 0.5:
                                    step_aucs.append(auc_s)
                            if len(step_losses) > 0:
                                loss_col = torch.stack(step_losses).mean()
                            if len(step_aucs) > 0:
                                pair_auc = torch.tensor(sum(step_aucs) / len(step_aucs), device=reward.device)
                            else:
                                pair_auc = torch.tensor(compute_pair_auc(H, actions_init[..., -1]), device=reward.device)
                        else:
                            act_target = actions_init[..., -1]
                            if self.lambda_col > 0.0:
                                loss_col = realized_collision_loss(H, act_target)
                            pair_auc = torch.tensor(compute_pair_auc(H, act_target), device=reward.device)
                    else:
                        if self.lambda_col > 0.0:
                            loss_col = realized_collision_loss(H, actions_init)
                        pair_auc = torch.tensor(compute_pair_auc(H, actions_init), device=reward.device)

                # 3. L_bal, L_sp
                if self.lambda_bal > 0.0 or self.lambda_sp > 0.0:
                    l_bal, l_sp = anti_collapse_losses(H)
                    if self.lambda_bal > 0.0:
                        loss_bal = l_bal
                    if self.lambda_sp > 0.0:
                        loss_sp = l_sp

                loss_aux = warmup_factor * (
                    self.lambda_int * loss_int
                    + self.lambda_col * loss_col
                    + self.lambda_bal * loss_bal
                    + self.lambda_sp * loss_sp
                )
                loss = loss + loss_aux

            # Hypergraph statistics & Gate logging
            gate_val = getattr(comm, "gate", torch.tensor(0.0)).item() if comm is not None and hasattr(comm, "gate") else 0.0
            prior_active = getattr(comm, "spatial_prior_active", False) if comm is not None else False

            if H is not None:
                deg = H.sum(dim=1)
                active_mask = (deg > 0.01)
                active_hyperedges_mean = active_mask.float().sum(dim=-1).mean()
                hyperedge_size_mean = deg[active_mask].mean() if active_mask.any() else torch.tensor(0.0, device=H.device)
                hyperedge_size_max = deg.max()
            else:
                active_hyperedges_mean = torch.tensor(0.0, device=reward.device)
                hyperedge_size_mean = torch.tensor(0.0, device=reward.device)
                hyperedge_size_max = torch.tensor(0.0, device=reward.device)

            aux_logs = {
                f"{phase}/loss_ss": loss_ss.item() if torch.is_tensor(loss_ss) else float(loss_ss),
                f"{phase}/loss_aux": loss_aux.item() if torch.is_tensor(loss_aux) else float(loss_aux),
                f"{phase}/loss_int": loss_int.item() if torch.is_tensor(loss_int) else float(loss_int),
                f"{phase}/loss_col": loss_col.item() if torch.is_tensor(loss_col) else float(loss_col),
                f"{phase}/loss_bal": loss_bal.item() if torch.is_tensor(loss_bal) else float(loss_bal),
                f"{phase}/loss_sp": loss_sp.item() if torch.is_tensor(loss_sp) else float(loss_sp),
                f"{phase}/pair_auc": pair_auc.item() if torch.is_tensor(pair_auc) else float(pair_auc),
                f"{phase}/gate_val": gate_val,
                f"{phase}/spatial_prior_active": float(prior_active),
                f"{phase}/active_hyperedges": active_hyperedges_mean.item() if torch.is_tensor(active_hyperedges_mean) else float(active_hyperedges_mean),
            }
            if hasattr(self, "log_dict"):
                try:
                    self.log_dict(aux_logs, on_step=(phase == "train"), on_epoch=True, prog_bar=False, sync_dist=True)
                except Exception:
                    pass

            out.update(
                {
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                    "loss_aux": loss_aux,
                    "loss_int": loss_int,
                    "loss_col": loss_col,
                    "loss_bal": loss_bal,
                    "loss_sp": loss_sp,
                    "pair_auc": pair_auc,
                    "gate_val": gate_val,
                    "spatial_prior_active": prior_active,
                    "active_hyperedges_mean": active_hyperedges_mean,
                    "hyperedge_size_mean": hyperedge_size_mean,
                    "hyperedge_size_max": hyperedge_size_max,
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {
            "loss": out.get("loss", None),
            "loss_ss": out.get("loss_ss", 0.0),
            "loss_aux": out.get("loss_aux", 0.0),
            "loss_int": out.get("loss_int", 0.0),
            "loss_col": out.get("loss_col", 0.0),
            "loss_bal": out.get("loss_bal", 0.0),
            "loss_sp": out.get("loss_sp", 0.0),
            "pair_auc": out.get("pair_auc", 0.5),
            "gate_val": out.get("gate_val", 0.0),
            "spatial_prior_active": out.get("spatial_prior_active", False),
            **metrics,
        }
