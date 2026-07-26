"""
Vehicles that live in the World.

`Vehicle` is the protocol the World steps against: anything with a ``position``,
a ``heading``, a ``size`` and a ``step(dt, world)`` method is a valid inhabitant.
Braitenberg's vehicles are *autonomous* -- each one reads the world through its
own senses and drives its own motors inside ``step`` -- so the World never pushes
control into them; it only hands them ``dt`` and a read-only view of itself.

`SimpleVehicle` is a placeholder implementation so the simulation runs end-to-end
today. It just cruises forward with a gentle constant turn. This ``step`` method
is the exact seam where real sensor->motor wiring (and, later, the neural-net
driven `entity.Entity`) will plug in.
"""
from typing import Protocol, Tuple, runtime_checkable

import numpy as np


def heading_to_vector(heading: float) -> np.ndarray:
    """Unit direction vector for a heading in radians (0 == +x, CCW positive)."""
    return np.array([np.cos(heading), np.sin(heading)], dtype=float)


@runtime_checkable
class Vehicle(Protocol):
    """Structural type for anything the World can hold and step.

    Implementations must expose a mutable 2D ``position`` (numpy array), a scalar
    ``heading`` in radians, an integer ``size`` pair, and a ``step`` that advances
    the vehicle by ``dt`` given a read-only ``world``.
    """
    position: np.ndarray
    heading: float
    size: Tuple[int, int]
    id: int

    def step(self, dt: float, world: "object") -> None: ...


class SimpleVehicle:
    """Minimal autonomous vehicle: constant speed, constant turn rate.

    Placeholder behaviour until senses/motors exist. Kept intentionally dumb so
    that ``step`` reads as the obvious place to insert Braitenberg wiring.
    """

    def __init__(
        self,
        position: Tuple[float, float],
        heading: float = 0.0,
        speed: float = 2.0,
        turn_rate: float = 0.03,
        size: Tuple[int, int] = (24, 24),
        id: int = 0,
    ):
        """
        Parameters
        ----------
        position : (x, y) start center in world coordinates
        heading : initial facing in radians (0 == +x, CCW positive)
        speed : world units advanced per unit dt
        turn_rate : radians added to heading per unit dt
        size : (width, height) used by the renderer
        id : identifier surfaced in the snapshot
        """
        self.position = np.asarray(position, dtype=float)
        self.heading = float(heading)
        self.speed = float(speed)
        self.turn_rate = float(turn_rate)
        self.size = size
        self.id = id

    def step(self, dt: float, world: "object") -> None:
        """Advance one tick. ``world`` is accepted (read-only) but unused for now."""
        self.heading += self.turn_rate * dt
        self.position = self.position + heading_to_vector(self.heading) * self.speed * dt
