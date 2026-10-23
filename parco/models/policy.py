from typing import Callable

import torch
import torch.nn as nn

from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from .agent_handlers import get_agent_handler
from .decoder import MatNetDecoder, PARCODecoder
from .decoding_strategies import (
    PARCO4FFSPDecoding,
    PARCODecodingStrategy,
    parco_get_decoding_strategy,
)
from .encoder import MatNetEncoder, PARCOEncoder
from .utils import get_log_likelihood

log = get_pylogger(__name__)


DEFAULTS_CONTEXT_KWARGS = {
    "use_communication": True,
    "num_communication_layers": 1,
    "normalization": "instance",
    "norm_after": False,
    "use_final_norm": False,
}


class PARCOPolicy(nn.Module):
    """Policy for PARCO model"""

    def __init__(
        self,
        encoder=None,
        decoder=None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        normalization: str = "instance",
        use_final_norm: bool = False,  # If True, normalize like in Llama
        norm_after: bool = False,
        env_name: str = "hcvrp",
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        agent_handler="highprob",  # Agent handler
        agent_handler_kwargs: dict = {},  # Agent handler kwargs
        use_init_logp: bool = True,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        replacement_value_key: str = "current_node",  # When stopping arises (conflict or POS token), replace the value of this key
        use_pos_token: bool = False,  # Add a POS (pause-of-sequence) action
        trainable_pos_token: bool = True,  # If true, then the pos token is trainable
        # two_stage_pos_sampling: bool = True,  # new, faster
        parallel_gated_kwargs: dict = None,  # ParallelGatedMLP kwargs
        sdpa_fn_decoder: (
            Callable | str
        ) = "simple",  # SDPA function for decoder, simple is JIC https://github.com/ai4co/rl4co/issues/228
    ):
        super(PARCOPolicy, self).__init__()

        if isinstance(agent_handler, str):
            agent_handler = get_agent_handler(agent_handler, **agent_handler_kwargs)
        self.agent_handler = agent_handler

        # If key is not provided, use default context kwargs
        context_embedding_kwargs = {
            **DEFAULTS_CONTEXT_KWARGS,
            **context_embedding_kwargs,
        }

        self.env_name = env_name

        # Encoder and decoder
        if encoder is None:
            log.info("Initializing default PARCOEncoder")
            self.encoder = PARCOEncoder(
                env_name=self.env_name,
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                init_embedding_kwargs=init_embedding_kwargs,
                use_final_norm=use_final_norm,
                norm_after=norm_after,
                use_pos_token=use_pos_token,
                trainable_pos_token=trainable_pos_token,
                parallel_gated_kwargs=parallel_gated_kwargs,
            )
        else:
            log.warning("Using custom encoder")
            self.encoder = encoder

        if decoder is None:
            log.info("Initializing default PARCODecoder")
            self.decoder = PARCODecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_embedding=context_embedding,
                context_embedding_kwargs=context_embedding_kwargs,
                env_name=self.env_name,
                use_pos_token=use_pos_token,
                sdpa_fn=sdpa_fn_decoder,
            )
        else:
            log.warning("Using custom decoder")
            self.decoder = decoder
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # Multi-agent handling
        self.replacement_value_key = replacement_value_key
        self.use_pos_token = use_pos_token
        # self.two_stage_pos_sampling = two_stage_pos_sampling
        self.mask_handled = mask_handled
        self.use_init_logp = use_init_logp

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        return_init_embeds: bool = True,
        **decoding_kwargs,
    ) -> dict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # When decode_type is sampling, we need to know the number of samples
        num_samples = decoding_kwargs.pop("num_samples", 1)
        if "sampling" not in decode_type:
            num_samples = 1

        # [B, m, N]
        num_agents = td["action_mask"].shape[-2]

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: PARCODecodingStrategy = parco_get_decoding_strategy(
            decode_type,
            num_agents=num_agents,
            agent_handler=self.agent_handler,
            use_init_logp=self.use_init_logp,
            mask_handled=self.mask_handled,
            replacement_value_key=self.replacement_value_key,
            num_samples=num_samples,
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_samples = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_samples)
        # We use unbatchify if num_samples > 1
        if num_samples > 1:
            do_unbatchify = True
        else:
            do_unbatchify = False
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            # We only need to proceed once when decoder forwarding.
            if step > 1:
                do_unbatchify = False

            logits, mask = self.decoder(td, hidden, num_samples, do_unbatchify)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]  # do not save the state
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) during decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        (
            logprobs,
            actions,
            td,
            env,
            halting_ratio,
        ) = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
            "halting_ratio": halting_ratio,
        }

        if return_actions:
            outdict["actions"] = actions
            outdict["actions_init"] = getattr(decode_strategy, "collected_actions_init", actions)

        if return_init_embeds:  # for SymNCO
            outdict["init_embeds"] = init_embeds

        outdict["steps"] = step  # Number of steps taken during decoding

        return outdict


class PARCOMultiStagePolicy(nn.Module):
    """Apply a OneStageModel for each stage"""

    def __init__(
        self,
        num_stages: int,
        embed_dim: int = 256,
        num_encoder_layers: int = 3,
        num_heads: int = 16,
        feedforward_hidden: int = 512,
        ms_hidden_dim: int = 32,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        dynamic_embedding: nn.Module = None,
        dynamic_embedding_kwargs: dict = {},
        normalization: str = "instance",
        norm_after: bool = True,
        scale_factor: float = 10,
        use_decoder_mha_mask: bool = False,
        use_ham_encoder: bool = True,
        pointer_check_nan: bool = True,
        env_name: str = "ffsp",
        use_pos_token: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
        agent_handler=None,  # Agent handler
        agent_handler_kwargs: dict = {},  # Agent handler kwargs
        use_init_logp: bool = False,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        replacement_value_key: str = "current_node",  # When stopping arises (conflict or POS token), replace the value of this key
        **transformer_kwargs,
    ):
        super().__init__()

        # Multi-agent handling
        self.use_pos_token = use_pos_token
        self.mask_handled = mask_handled
        self.use_init_logp = use_init_logp

        if agent_handler is not None:
            if isinstance(agent_handler, str):
                agent_handler = get_agent_handler(agent_handler, **agent_handler_kwargs)
        self.agent_handler = agent_handler

        self.stage_cnt = num_stages
        self.stage_models = nn.ModuleList(
            [
                OneStageModel(
                    stage_idx,
                    num_stages,
                    embed_dim=embed_dim,
                    num_encoder_layers=num_encoder_layers,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    ms_hidden_dim=ms_hidden_dim,
                    init_embedding=init_embedding,
                    init_embedding_kwargs=init_embedding_kwargs,
                    context_embedding=context_embedding,
                    context_embedding_kwargs=context_embedding_kwargs,
                    dynamic_embedding=dynamic_embedding,
                    dynamic_embedding_kwargs=dynamic_embedding_kwargs,
                    normalization=normalization,
                    norm_after=norm_after,
                    scale_factor=scale_factor,
                    env_name=env_name,
                    use_pos_token=use_pos_token,
                    use_decoder_mha_mask=use_decoder_mha_mask,
                    use_ham_encoder=use_ham_encoder,
                    pointer_check_nan=pointer_check_nan,
                    agent_handler=agent_handler,
                    agent_handler_kwargs=agent_handler_kwargs,
                    use_init_logp=use_init_logp,
                    mask_handled=mask_handled,
                )
                for stage_idx in range(self.stage_cnt)
            ]
        )

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def pre_forward(self, td: TensorDict, env, num_starts: int, decode_type):
        # exclude the dummy node and split into stage tables
        n_agents = td["job_duration"].size(-1)
        num_jobs = td["job_duration"].size(-2) - 1
        run_time_list = td["job_duration"][:, :-1].chunk(env.num_stage, dim=-1)
        for stage_idx in range(self.stage_cnt):
            td["cost_matrix"] = run_time_list[stage_idx]
            model: OneStageModel = self.stage_models[stage_idx]
            model.pre_forward(td, env, num_starts)

        if num_starts > 1:
            # repeat num_start times
            td = batchify(td, num_starts)

        # update machine idx and action mask
        td = env.pre_step(td)

        self.decode_strategy: PARCODecodingStrategy = parco_get_decoding_strategy(
            decode_type,
            num_agents=n_agents,
            agent_handler=self.agent_handler,
            use_init_logp=self.use_init_logp,
            mask_handled=self.mask_handled,
            replacement_value_key="wait_action",
            tanh_clipping=10,
        )

        self.decode_strategy = PARCO4FFSPDecoding(
            num_ma=n_agents, num_job=num_jobs, tanh_clipping=10, use_pos_token=True
        )

        return td

    def forward(
        self,
        td: TensorDict,
        env,
        phase="train",
        num_starts=1,
        return_actions: bool = False,
        return_sum_log_likelihood: bool = True,
        **decoder_kwargs,
    ):
        # Get decode type depending on phase
        decode_type = getattr(self, f"{phase}_decode_type")

        td = self.pre_forward(td, env, num_starts, decode_type)

        # NOTE: this must come after pre_forward due to batchify op
        # collect some data statistics
        bs, total_mas, num_jobs_plus_one = td["full_action_mask"].shape
        num_jobs = num_jobs_plus_one - 1
        n_stage_mas = total_mas // self.stage_cnt

        steps = 0
        while not td["done"].all():
            action_stack = []
            for stage_idx in range(self.stage_cnt):
                model = self.stage_models[stage_idx]
                action_mask_list = td["full_action_mask"].chunk(self.stage_cnt, dim=1)
                td["action_mask"] = action_mask_list[stage_idx]
                logits = model(td)
                td["wait_action"] = torch.full(
                    (bs, n_stage_mas),
                    fill_value=num_jobs,
                    dtype=torch.long,
                    device=td.device,
                )
                td = self.decode_strategy.step(
                    logits,
                    td["action_mask"],
                    td,
                    agent_handler_kwargs={"randomize": self.training},
                )
                action = td.pop("action")
                action_stack.append(action)

            action_stack = torch.cat(action_stack, dim=-1)
            td.set("action", action_stack)
            td = env.step(td)["next"]
            steps += 1

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        (
            logprobs,
            actions,
            td,
            env,
            halting_ratio,
        ) = self.decode_strategy.post_decoder_hook(td, env)

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs,
                actions=None,
                mask=td.get("mask", None),
                return_sum=return_sum_log_likelihood,
            ).sum(1),
            "halting_ratio": halting_ratio,
        }

        if return_actions:
            outdict["actions"] = actions

        outdict["steps"] = steps  # Number of steps taken during decoding

        return outdict


class OneStageModel(nn.Module):
    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        embed_dim: int = 256,
        num_encoder_layers: int = 3,
        num_heads: int = 16,
        feedforward_hidden: int = 512,
        ms_hidden_dim: int = 32,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        dynamic_embedding: nn.Module = None,
        dynamic_embedding_kwargs: dict = {},
        normalization: str = "instance",
        norm_after: bool = False,
        scale_factor: float = 100,
        env_name: str = "ffsp",
        use_pos_token: bool = True,
        use_decoder_mha_mask: bool = False,
        use_ham_encoder: bool = True,
        pointer_check_nan: bool = True,
        agent_handler=None,  # Agent handler
        agent_handler_kwargs: dict = {},  # Agent handler kwargs
        use_init_logp: bool = True,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        **transformer_kwargs,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        if agent_handler is not None:
            if isinstance(agent_handler, str):
                agent_handler = get_agent_handler(agent_handler, **agent_handler_kwargs)
        self.agent_handler = agent_handler

        self.encoder = MatNetEncoder(
            stage_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            ms_hidden_dim=ms_hidden_dim,
            normalization=normalization,
            init_embedding=init_embedding,
            init_embedding_kwargs=init_embedding_kwargs,
            scale_factor=scale_factor,
            use_ham=use_ham_encoder,
            norm_after=norm_after,
            **transformer_kwargs,
        )

        self.decoder = MatNetDecoder(
            stage_idx,
            num_stages,
            embed_dim=embed_dim,
            num_heads=num_heads,
            scale_factor=scale_factor,
            env_name=env_name,
            context_embedding=context_embedding,
            context_embedding_kwargs=context_embedding_kwargs,
            dynamic_embedding=dynamic_embedding,
            dynamic_embedding_kwargs=dynamic_embedding_kwargs,
            mask_inner=use_decoder_mha_mask,
            check_nan=pointer_check_nan,
        )

        # Multi-agent handling
        self.use_pos_token = use_pos_token
        self.mask_handled = mask_handled
        self.use_init_logp = use_init_logp

    def pre_forward(self, td, env, num_starts):
        embeddings = self.encoder(td)
        # encoded_row.shape: (batch, job_cnt, embedding)
        # encoded_col.shape: (batch, machine_cnt, embedding)
        td, env, cached = self.decoder.pre_decoder_hook(
            td, env, embeddings, num_starts=num_starts
        )
        self.cache = cached

        self.num_job = embeddings[0].size(1)
        self.num_ma = embeddings[1].size(1)

    def forward(self, td: TensorDict, num_starts: int = 0):
        # shape: (batch, num_agents, job_cnt+1)
        logits, _ = self.decoder(td, self.cache, num_starts=num_starts)
        return logits
