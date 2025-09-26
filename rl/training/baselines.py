"""Baseline optimizers such as CMA-ES for preset search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import cma
except ImportError:  # pragma: no cover - optional dependency
    cma = None


ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class CmaConfig:
    sigma0: float = 0.2
    max_iters: int = 400
    population_size: int | None = None


class CmaEsBaseline:
    """Runs CMA-ES over normalized parameter vectors."""

    def __init__(self, dimension: int, config: CmaConfig | None = None) -> None:
        if cma is None:
            raise ImportError("cma package required for CMA-ES baseline. Install with `pip install cma`." )
        self.dimension = dimension
        self.config = config or CmaConfig()

    def optimize(self, objective: ObjectiveFn, initial: np.ndarray | None = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float]]]:
        if initial is None:
            initial = 0.5 * np.ones(self.dimension, dtype=np.float32)
        opts = {
            "seed": 0,
            "popsize": self.config.population_size,
            "maxiter": self.config.max_iters,
        }
        es = cma.CMAEvolutionStrategy(initial, self.config.sigma0, opts)
        history: List[Tuple[np.ndarray, float]] = []
        while not es.stop():
            solutions = es.ask()
            values = [objective(np.clip(np.array(s, dtype=np.float32), 0.0, 1.0)) for s in solutions]
            es.tell(solutions, values)
            best_idx = int(np.argmin(values))
            history.append((solutions[best_idx], values[best_idx]))
        return np.clip(es.result.xbest, 0.0, 1.0), history
