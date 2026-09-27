import torch
import torch.nn.functional as F


def affinity_from_history(
    history: torch.Tensor, knn: int = 8, sigma: float = 1.0
) -> torch.Tensor:
    """Construct a symmetric normalized graph Laplacian from agent state histories.

    Args:
        history: Tensor of shape [B, M, W, d]
        knn: Number of nearest neighbors for affinity graph
        sigma: Kernel bandwidth parameter

    Returns:
        Symmetric normalized Laplacian matrix of shape [B, M, M]
    """
    batch_size, num_agents = history.shape[:2]
    if num_agents <= 1:
        return torch.zeros(batch_size, num_agents, num_agents, device=history.device, dtype=history.dtype)

    # Flatten history window and normalize
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


def batched_kmeans(
    embedding: torch.Tensor, num_clusters: int, steps: int = 5
) -> torch.Tensor:
    """Deterministic batched k-means algorithm for spectral cluster assignment.

    Args:
        embedding: Spectral embedding tensor [B, M, k]
        num_clusters: Number of target clusters
        steps: Number of k-means iterations

    Returns:
        Tensor of shape [B, M] with integer cluster assignments in [0, num_clusters - 1]
    """
    batch_size, num_agents, _ = embedding.shape
    num_clusters = min(num_clusters, num_agents)

    if num_clusters <= 1:
        return torch.zeros(
            batch_size, num_agents, dtype=torch.long, device=embedding.device
        )

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
