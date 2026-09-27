import abc
import logging
import torch

from .config import HSCLConfig

log = logging.getLogger(__name__)


class EigenSolverBackend(abc.ABC):
    """Abstract base class for eigensolver backends."""

    @abc.abstractmethod
    def smallest_k_eigenvectors(self, L: torch.Tensor, k: int) -> torch.Tensor:
        """Compute the eigenvectors corresponding to the k smallest eigenvalues.

        Args:
            L: Symmetric normalized Laplacian tensor of shape [B, M, M]
            k: Number of eigenvectors to compute

        Returns:
            Tensor of shape [B, M, k] containing the smallest k eigenvectors.
        """
        raise NotImplementedError


class DenseGPUEigh(EigenSolverBackend):
    """Dense eigensolver using torch.linalg.eigh on GPU."""

    def smallest_k_eigenvectors(self, L: torch.Tensor, k: int) -> torch.Tensor:
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigvecs = torch.linalg.eigh(L).eigenvectors
        return eigvecs[..., :k]


class DenseCPUEigh(EigenSolverBackend):
    """Dense eigensolver offloaded to CPU to bypass cuSOLVER's non-batched fallback."""

    def smallest_k_eigenvectors(self, L: torch.Tensor, k: int) -> torch.Tensor:
        orig_device = L.device
        L_cpu = L.detach().to("cpu", non_blocking=False)
        M = L.size(-1)
        eps = torch.finfo(L.dtype).eps * 10
        L_cpu = L_cpu + eps * torch.eye(M, dtype=L_cpu.dtype)

        # Setting num_threads=1 for small 50x50 matrices avoids OpenMP synchronization overhead
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            eigvecs = torch.linalg.eigh(L_cpu).eigenvectors[..., :k]
        finally:
            torch.set_num_threads(orig_threads)

        return eigvecs.to(orig_device, non_blocking=False)


class LOBPCGApprox(EigenSolverBackend):
    """Iterative eigensolver using torch.lobpcg with fallback to DenseCPUEigh."""

    def __init__(self, niter: int = 15):
        self.niter = niter
        self._fallback = DenseCPUEigh()

    def smallest_k_eigenvectors(self, L: torch.Tensor, k: int) -> torch.Tensor:
        batch_size, M = L.shape[:2]
        # Regularize to guarantee positive definiteness for LOBPCG
        L_reg = L + 1e-4 * torch.eye(M, device=L.device, dtype=L.dtype).expand(batch_size, -1, -1)
        X = torch.randn(batch_size, M, k, device=L.device, dtype=L.dtype)
        try:
            _, V = torch.lobpcg(L_reg, k=k, largest=False, X=X, niter=self.niter)
            return V
        except Exception as e:
            log.warning(f"LOBPCG failed to converge ({e}), falling back to DenseCPUEigh.")
            return self._fallback.smallest_k_eigenvectors(L, k)


class AutoEigenSolver(EigenSolverBackend):
    """Swappable Eigensolver with automatic hardware-aware dispatching."""

    def __init__(self, config: HSCLConfig = None):
        self.config = config or HSCLConfig()
        self.dense_gpu = DenseGPUEigh()
        self.dense_cpu = DenseCPUEigh()
        self.lobpcg = LOBPCGApprox()

    def smallest_k_eigenvectors(self, L: torch.Tensor, k: int) -> torch.Tensor:
        M = L.size(-1)
        k = min(k, M)
        backend = self.config.eigensolver_backend

        if backend == "dense_gpu":
            return self.dense_gpu.smallest_k_eigenvectors(L, k)
        elif backend == "dense_cpu":
            return self.dense_cpu.smallest_k_eigenvectors(L, k)
        elif backend == "lobpcg":
            return self.lobpcg.smallest_k_eigenvectors(L, k)
        elif backend == "auto":
            if L.is_cuda and M <= self.config.dense_gpu_safe_max_agents:
                return self.dense_gpu.smallest_k_eigenvectors(L, k)
            else:
                # For M > 32 on GPU (or CPU inputs), use DenseCPUEigh to avoid cuSOLVER's cliff
                return self.dense_cpu.smallest_k_eigenvectors(L, k)
        else:
            raise ValueError(f"Unknown eigensolver backend: {backend}")
