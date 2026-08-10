"""
Serializable snapshot of the simulation, passed from the model (World) to the
view (Renderer) and to any logging / training consumers.

These are plain, immutable data objects: they hold no pygame handles and no live
references back into the World, so a snapshot is safe to render, log, or pickle
without worrying about mutation mid-frame.
"""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class EntityState:
    """Immutable pose of a single entity at one tick."""
    id: int
    position: Tuple[float, float]   # (x, y) center, world coordinates
    facing_point: Tuple[float, float]
    size: Tuple[int, int]           # (width, height) for drawing
    sense_poly: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class WorldState:
    """Immutable snapshot of the whole world at one tick."""
    tick: int
    width: int
    height: int
    entities: Tuple[EntityState, ...]
