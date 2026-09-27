from dataclasses import dataclass


@dataclass
class HSCLConfig:
    """Configuration for Hypergraph-Structured Communication Layer (HSCL)."""

    use_hscl: bool = True
    history_window: int = 8
    knn: int = 8
    sigma: float = 1.0
    num_clusters: int = 4
    recluster_every_k_steps: int = 1
    eigensolver_backend: str = "auto"  # "auto" | "dense_gpu" | "dense_cpu" | "lobpcg"
    dense_gpu_safe_max_agents: int = 32  # Exact threshold from hardware benchmark
    lambda_group_loss: float = 0.0
    inter_group_top_r: int = 2
    recluster_per_layer: bool = False
    ffsp_cluster_scope: str = "per_stage"  # "per_stage" | "global"
