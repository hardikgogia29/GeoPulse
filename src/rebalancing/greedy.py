
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Move:
    timestamp: object
    source: str
    destination: str
    bikes: int
    distance_km: float


@dataclass
class SimulationResult:
    unserved_pickups: float = 0.0
    overflow_dropoffs: float = 0.0
    empty_intervals: int = 0
    full_intervals: int = 0
    served_pickups: float = 0.0
    total_pickup_demand: float = 0.0
    bikes_moved: int = 0
    distance_km: float = 0.0
    moves: list[Move] = field(default_factory=list)

    @property
    def service_level(self) -> float:
        """Share of pickup demand actually satisfiable from stock on hand."""
        if self.total_pickup_demand == 0:
            return float("nan")
        return self.served_pickups / self.total_pickup_demand

    def summary(self) -> dict:
        return {
            "service_level": self.service_level,
            "unserved_pickups": self.unserved_pickups,
            "overflow_dropoffs": self.overflow_dropoffs,
            "empty_intervals": self.empty_intervals,
            "full_intervals": self.full_intervals,
            "bikes_moved": self.bikes_moved,
            "distance_km": round(self.distance_km, 1),
            "n_moves": len(self.moves),
        }


def project_inventory(inventory: np.ndarray, predicted_pickups: np.ndarray,
                      predicted_dropoffs: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Inventory implied at the end of the forecast horizon, clipped to the docks."""
    projected = inventory + predicted_dropoffs.cumsum(axis=0)[-1] \
        - predicted_pickups.cumsum(axis=0)[-1]
    return np.clip(projected, 0, capacity)


def shortage_surplus(projected: np.ndarray, capacity: np.ndarray,
                     safety_pct: float, target_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Bikes short of the safety floor, and spare above the target level."""
    safety = safety_pct * capacity
    target = target_pct * capacity
    shortage = np.maximum(safety - projected, 0.0)
    surplus = np.maximum(projected - target, 0.0)
    return shortage, surplus


def greedy_moves(shortage: np.ndarray, surplus: np.ndarray, inventory: np.ndarray,
                 capacity: np.ndarray, distances: np.ndarray, region_ids: list[str],
                 timestamp, max_moves: int = 200,
                 max_distance_km: float = 5.0) -> list[Move]:
    """Largest shortage first, matched to the nearest region with spare bikes.

    Constraints enforced on every move: never take more than the source's surplus,
    never exceed the destination's free dock space, never drive either side negative.
    """
    shortage = shortage.copy()
    surplus = surplus.copy()
    inventory = inventory.copy()
    moves: list[Move] = []

    for _ in range(max_moves):
        destination = int(np.argmax(shortage))
        if shortage[destination] <= 0.5:
            break
        candidates = np.flatnonzero(surplus > 0.5)
        if candidates.size == 0:
            break
        reachable = candidates[distances[destination, candidates] <= max_distance_km]
        if reachable.size == 0:
            shortage[destination] = 0.0  # nothing in range; stop chasing this one
            continue
        source = int(reachable[np.argmin(distances[destination, reachable])])

        free_docks = capacity[destination] - inventory[destination]
        bikes = int(min(shortage[destination], surplus[source],
                        inventory[source], free_docks))
        if bikes <= 0:
            surplus[source] = 0.0
            continue

        inventory[source] -= bikes
        inventory[destination] += bikes
        surplus[source] -= bikes
        shortage[destination] -= bikes
        moves.append(Move(timestamp, region_ids[source], region_ids[destination],
                          bikes, float(distances[destination, source])))
    return moves


def simulate(inventory0: np.ndarray, actual_pickups: np.ndarray,
             actual_dropoffs: np.ndarray, capacity: np.ndarray,
             moves_by_step: dict[int, list[tuple[int, int, int]]] | None = None
             ) -> SimulationResult:
    """Step the network through actual demand, optionally applying rebalancing moves.

    Unserved pickups are demand arriving at an empty region; overflow dropoffs are
    returns arriving at a full one. Both are counted rather than silently clipped -
    they are the whole point of the exercise.
    """
    inventory = inventory0.astype(np.float64).copy()
    result = SimulationResult()
    steps = actual_pickups.shape[0]

    for step in range(steps):
        if moves_by_step and step in moves_by_step:
            for source, destination, bikes in moves_by_step[step]:
                bikes = int(min(bikes, inventory[source],
                                capacity[destination] - inventory[destination]))
                if bikes > 0:
                    inventory[source] -= bikes
                    inventory[destination] += bikes
                    result.bikes_moved += bikes

        demand = actual_pickups[step].astype(np.float64)
        served = np.minimum(demand, inventory)
        unserved = demand - served
        inventory -= served

        returns = actual_dropoffs[step].astype(np.float64)
        space = capacity - inventory
        accepted = np.minimum(returns, space)
        overflow = returns - accepted
        inventory += accepted

        result.total_pickup_demand += float(demand.sum())
        result.served_pickups += float(served.sum())
        result.unserved_pickups += float(unserved.sum())
        result.overflow_dropoffs += float(overflow.sum())
        result.empty_intervals += int((inventory <= 0).sum())
        result.full_intervals += int((inventory >= capacity).sum())

    return result
