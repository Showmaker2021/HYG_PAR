import torch
from typing import Optional


class StateHistoryBuffer:
    """Maintains a sliding window of historical agent representations across decoding steps."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.buffer: Optional[torch.Tensor] = None

    def reset(self):
        """Clear buffer state."""
        self.buffer = None

    def update(self, context: torch.Tensor) -> torch.Tensor:
        """Append the current context embedding to the historical buffer.

        Args:
            context: Current step agent embedding of shape [B, M, d]

        Returns:
            Updated history tensor of shape [B, M, W, d]
        """
        B, M, d = context.shape
        if (
            self.buffer is None
            or self.buffer.shape[0] != B
            or self.buffer.shape[1] != M
            or self.buffer.device != context.device
        ):
            # Initial fill: replicate initial context across the entire window
            self.buffer = context.unsqueeze(2).expand(-1, -1, self.window_size, -1).clone()
        else:
            # Slide window: pop oldest entry and append new context
            self.buffer = torch.cat((self.buffer[:, :, 1:], context.unsqueeze(2)), dim=2)
        return self.buffer
