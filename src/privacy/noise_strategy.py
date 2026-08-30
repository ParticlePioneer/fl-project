"""
Privacy noise strategy interface.

This module decouples "how much privacy noise to apply" from "how training
happens." Right now only FixedNoiseStrategy is implemented (same epsilon for
every client, every round). The interface is built so an AdaptiveNoiseStrategy
can be added later (e.g. per-client epsilon scaled by dataset size, or
epsilon that changes across rounds) without touching client.py or the Opacus
integration itself -- only a new subclass needs to be written and swapped in
via config.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ClientMetadata:
    """
    Metadata passed to a NoiseStrategy so it can make per-client decisions.

    Populated by client.py on every round, even though FixedNoiseStrategy
    ignores all of it for now. Keeping this plumbing in place from day one
    means AdaptiveNoiseStrategy can be dropped in later without changing
    client.py's call signature.
    """
    client_id: str
    round_num: int
    dataset_size: int
    extra: Dict[str, Any] = field(default_factory=dict)


class NoiseStrategy(ABC):
    """Abstract base for any policy that decides the privacy budget (epsilon)
    used by a client at a given point in training."""

    @abstractmethod
    def get_epsilon(self, metadata: ClientMetadata) -> float:
        """Return the epsilon value to use for this client at this round."""
        raise NotImplementedError

    def get_delta(self, metadata: ClientMetadata) -> float:
        """
        Return the delta value (DP failure probability) to use.
        Default: 1 / dataset_size, a common convention. Override if needed.
        """
        return 1.0 / max(metadata.dataset_size, 1)


class FixedNoiseStrategy(NoiseStrategy):
    """
    The strategy used for the entire 8-week core project.
    Every client, every round, gets the same epsilon -- set via config.
    """

    def __init__(self, epsilon: float, delta: Optional[float] = None):
        self.epsilon = epsilon
        self._delta = delta

    def get_epsilon(self, metadata: ClientMetadata) -> float:
        return self.epsilon

    def get_delta(self, metadata: ClientMetadata) -> float:
        if self._delta is not None:
            return self._delta
        return super().get_delta(metadata)


# ---------------------------------------------------------------------------
# Placeholder for future extension. Not implemented in the 8-week scope.
# Documented here intentionally so the architectural intent is explicit for
# anyone (including future-you) reading the repo.
# ---------------------------------------------------------------------------
# class AdaptiveNoiseStrategy(NoiseStrategy):
#     """
#     Example future direction: scale epsilon per client based on dataset
#     size (smaller clients get a larger/looser epsilon budget since they
#     have less data to protect proportionally, or vice versa depending on
#     the chosen adaptive scheme -- to be decided if/when this is built).
#     """
#     def __init__(self, base_epsilon: float, scaling_method: str):
#         self.base_epsilon = base_epsilon
#         self.scaling_method = scaling_method
#
#     def get_epsilon(self, metadata: ClientMetadata) -> float:
#         raise NotImplementedError("Stretch goal -- see decisions.md")


def build_noise_strategy(config: dict) -> NoiseStrategy:
    """
    Factory function: reads the 'privacy' section of the config and returns
    the configured NoiseStrategy instance. This is the single place that
    needs to change to support new strategies -- client.py and server.py
    never need to know which concrete strategy is in use.
    """
    strategy_name = config.get("strategy", "fixed")

    if strategy_name == "fixed":
        fixed_cfg = config.get("fixed", {})
        return FixedNoiseStrategy(
            epsilon=fixed_cfg.get("epsilon", 4.0),
            delta=fixed_cfg.get("delta", None),
        )

    raise ValueError(
        f"Unknown privacy strategy '{strategy_name}'. "
        f"Only 'fixed' is implemented in the current scope."
    )