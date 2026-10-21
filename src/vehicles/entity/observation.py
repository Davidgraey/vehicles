"""
What a single vehicle knows within one tick.
These are plain, immutable snapshots: nothing here holds a live reference to
another entity, so a vehicle can only act on what it was able to observe.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from vehicles.entity.angles import Angle


@dataclass(frozen=True)
class Detection:
    """
    One sensed target, as seen from the observing vehicle.

    Parameters
    ----------
    position : (x, y) world coordinates
    strength : signal strength at the sensor
    kind : class of the detected entity
    velocity : estimated speed of the target, None until estimation exists
    heading : estimated heading of the target, None until estimation exists
    """
    position: Tuple[float, float]
    strength: float
    kind: type

    # TODO: mechanism to estimate target velocity and heading from successive observations
    velocity: Optional[float] = None
    heading: Optional[Angle] = None


@dataclass(frozen=True)
class Observation:
    """
    Everything one vehicle is able to observe this tick.

    Parameters
    ----------
    detections : sensed entities
    terrain : sensed terrain, at most one detection per terrain region, at its nearest sensed point
    """
    detections: Tuple[Detection, ...]
    terrain: Tuple[Detection, ...] = ()
