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
    has_detections: bool = False
    is_colliding: bool = False
    color: str = "steelblue"
    body_poly: Optional[Tuple[int, int]] = None
    hunger: Optional[float] = None
    max_hunger: Optional[float] = None
    food: Optional[float] = None
    max_food: Optional[float] = None
    health: float = 100.0
    max_health: float = 100.0
    active_behavior: Optional[str] = None
    is_controlled: bool = False


@dataclass(frozen=True)
class TerrainState:
    """
    Immutable description of one terrain region.

    Parameters
    ----------
    color : draw color
    points : (x, y) points of a closed boundary, or of a polyline
    thickness : stroke width of a polyline; 0 for a closed region
    closed : True when `points` is a closed boundary
    """
    color: str
    points: Tuple[Tuple[float, float], ...]
    thickness: float = 0.0
    closed: bool = True


@dataclass(frozen=True)
class WorldState:
    """Immutable snapshot of the whole world at one tick."""
    tick: int
    width: int
    height: int
    entities: Tuple[EntityState, ...]
    terrain: Tuple[TerrainState, ...] = ()
