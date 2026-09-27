"""Profile the computational cost of the proposed HSCL ingredients.

This is a Phase-0 benchmark: it does not train a policy or change PARCO.
For each environment and agent count it measures the existing context and
decoder calls alongside a spectral-grouping pipeline operating on the real
context embeddings produced by that environment.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parco.envs.ffsp.env import FFSPEnv
from parco.envs.hcvrp.env import HCVRPEnv
from parco.envs.omdcpdp.env import OMDCPDPEnv
from parco.models.policy import PARCOMultiStagePolicy, PARCOPolicy
from parco.models.hscl import AutoEigenSolver, HSCLConfig


@dataclass
class Measurement:
    name: str
    mean_ms: float
    std_ms: float


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> Measurement:
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1_000)

    return Measurement(
        name="",
        mean_ms=statistics.mean(samples),
        std_ms=statistics.pstdev(samples),
    )


def update_history(history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Append a context vector to a fixed-size intra-instance state buffer."""
    return torch.cat((history[:, :, 1:], context.unsqueeze(2)), dim=2)


def affinity_from_history(history: torch.Tensor, knn: int, sigma: float) -> torch.Tensor:
    """Construct a symmetric normalized graph Laplacian from agent histories."""
    batch_size, num_agents = history.shape[:2]
    states = F.normalize(history.flatten(2), dim=-1)
    similarity = states @ states.transpose(-1, -2)
    similarity.diagonal(dim1=-2, dim2=-1).fill_(-torch.inf)

    neighbor_count = min(knn, max(num_agents - 1, 1))
    scores, indices = similarity.topk(neighbor_count, dim=-1)
    weights = torch.exp((scores - 1.0) / (sigma * sigma))
    adjacency = torch.zeros_like(similarity)
    adjacency.scatter_(-1, indices, weights)
    adjacency = torch.maximum(adjacency, adjacency.transpose(-1, -2))

    degree = adjacency.sum(-1).clamp_min(torch.finfo(adjacency.dtype).eps)
    inv_sqrt_degree = degree.rsqrt()
    normalized = (
        adjacency
        * inv_sqrt_degree.unsqueeze(-1)
        * inv_sqrt_degree.unsqueeze(-2)
    )
    identity = torch.eye(num_agents, device=history.device, dtype=history.dtype)
    return identity.expand(batch_size, -1, -1) - normalized


def batched_kmeans(embedding: torch.Tensor, num_clusters: int, steps: int = 5) -> torch.Tensor:
    """Small deterministic batched k-means used only to time spectral assignment."""
    batch_size, num_agents, _ = embedding.shape
    num_clusters = min(num_clusters, num_agents)
    seed_indices = torch.linspace(
        0, num_agents - 1, steps=num_clusters, device=embedding.device
    ).long()
    centroids = embedding[:, seed_indices].clone()
    assignment = torch.zeros(
        batch_size, num_agents, dtype=torch.long, device=embedding.device
    )
    for _ in range(steps):
        distances = (embedding.unsqueeze(2) - centroids.unsqueeze(1)).square().sum(-1)
        assignment = distances.argmin(-1)
        membership = F.one_hot(assignment, num_classes=num_clusters).to(embedding.dtype)
        counts = membership.sum(1).clamp_min(1).unsqueeze(-1)
        centroids = torch.einsum("bmk,bmd->bkd", membership, embedding) / counts
    return assignment


def routing_functions(
    problem: str, num_agents: int, batch_size: int, num_loc: int, device: torch.device
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    env_class = HCVRPEnv if problem == "hcvrp" else OMDCPDPEnv
    env = env_class(generator_params={"num_loc": num_loc, "num_agents": num_agents})
    td = env.reset(batch_size=[batch_size]).to(device)
    policy = PARCOPolicy(env_name=problem, agent_handler="highprob").to(device).eval()
    with torch.inference_mode():
        hidden, _ = policy.encoder(td)
        td, env, cached = policy.decoder.pre_decoder_hook(td, env, hidden, num_starts=1)

    def context() -> torch.Tensor:
        return policy.decoder.context_embedding(cached.node_embeddings, td)

    def decoder() -> torch.Tensor:
        return policy.decoder(td, cached, num_starts=1, do_unbatchify=False)[0]

    return context, decoder


def ffsp_functions(
    num_agents: int, batch_size: int, num_jobs: int, device: torch.device
) -> tuple[Callable[[], torch.Tensor], Optional[Callable[[], torch.Tensor]]]:
    num_stages = 3
    env = FFSPEnv(
        generator_params={
            "num_stage": num_stages,
            "num_machine": num_agents,
            "num_job": num_jobs,
        }
    )
    td = env.reset(batch_size=[batch_size]).to(device)
    policy = PARCOMultiStagePolicy(
        num_stages=num_stages,
        embed_dim=256,
        num_heads=16,
        feedforward_hidden=512,
        init_embedding_kwargs={"one_hot_seed_cnt": num_agents},
        context_embedding_kwargs={
            "use_comm_layer": True,
            "num_heads": 16,
            "normalization": "instance",
            "feedforward_hidden": 512,
        },
        dynamic_embedding_kwargs={"scale_factor": 10},
    ).to(device).eval()
    with torch.inference_mode():
        td = policy.pre_forward(td, env, num_starts=1, decode_type="sampling")
        stage_model = policy.stage_models[0]
        td["action_mask"] = td["full_action_mask"].chunk(num_stages, dim=1)[0]

    def context() -> torch.Tensor:
        return stage_model.decoder.context_embedding(stage_model.cache.node_embeddings, td)

    # The current FFSP decoder cannot be benchmarked end-to-end: its cache includes
    # a dummy wait-job while FFSPDynamicEmbedding returns only real jobs. Keep the
    # Phase-0 measurement honest by timing its real communication layer only.
    return context, None


def profile_case(args: argparse.Namespace, problem: str, num_agents: int) -> dict:
    if problem == "ffsp":
        context_fn, decoder_fn = ffsp_functions(
            num_agents, args.batch_size, args.num_jobs, args.device
        )
    else:
        context_fn, decoder_fn = routing_functions(
            problem, num_agents, args.batch_size, args.num_loc, args.device
        )

    with torch.inference_mode():
        context = context_fn()
        history = context.unsqueeze(2).expand(-1, -1, args.window, -1).contiguous()
        laplacian = affinity_from_history(history, args.knn, args.sigma)
        solver = AutoEigenSolver(
            HSCLConfig(
                eigensolver_backend=args.solver,
                dense_gpu_safe_max_agents=32,
            )
        )
        eigvecs = solver.smallest_k_eigenvectors(laplacian, args.num_clusters)

        functions = {
            "context_full_attention": context_fn,
            "history_update": lambda: update_history(history, context),
            "history_to_laplacian": lambda: affinity_from_history(
                history, args.knn, args.sigma
            ),
            "laplacian_eigh": lambda: solver.smallest_k_eigenvectors(
                laplacian, args.num_clusters
            ),
            "spectral_assignment": lambda: batched_kmeans(eigvecs, args.num_clusters),
        }
        if decoder_fn is not None:
            functions["decoder_forward"] = decoder_fn
        measurements = {}
        for name, function in functions.items():
            result = measure(function, args.device, args.warmup, args.iterations)
            result.name = name
            measurements[name] = asdict(result)

    context_ms = measurements["context_full_attention"]["mean_ms"]
    decoder_ms = measurements.get("decoder_forward", {}).get("mean_ms")
    spectral_ms = (
        measurements["history_to_laplacian"]["mean_ms"]
        + measurements["laplacian_eigh"]["mean_ms"]
        + measurements["spectral_assignment"]["mean_ms"]
    )
    return {
        "problem": problem,
        "num_agents": num_agents,
        "batch_size": args.batch_size,
        "context_shape": list(context.shape),
        "measurements": measurements,
        "spectral_pipeline_ms": spectral_ms,
        "spectral_over_context_percent": 100 * spectral_ms / context_ms,
        "spectral_over_decoder_percent": (
            100 * spectral_ms / decoder_ms if decoder_ms is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="hcvrp,omdcpdp,ffsp")
    parser.add_argument("--agents", default="5,10,20,30,50,60")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-loc", type=int, default=100)
    parser.add_argument("--num-jobs", type=int, default=20)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "dense_gpu", "dense_cpu", "lobpcg"],
        help="Eigensolver backend to use.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional JSON path for the complete measurements.")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


def main() -> None:
    args = parse_args()
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(1234)
    results = []
    for problem in args.problems.split(","):
        for num_agents in (int(value) for value in args.agents.split(",")):
            results.append(profile_case(args, problem, num_agents))

    report = {
        "device": str(args.device),
        "torch_version": torch.__version__,
        "solver": args.solver,
        "window": args.window,
        "knn": args.knn,
        "num_clusters": args.num_clusters,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
