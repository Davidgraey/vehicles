"""
Contact resolution shared by entity-entity and entity-terrain collisions.

Two touching bodies are separated along the contact normal, split by inverse mass, and exchange velocity along that normal. Terrain is a body of infinite
mass.
This is not a great long-term beahvior and we should revist this in the future.
"""
from typing import Optional

import numpy as np

from vehicles.entity.angles import (
    angular_motion_to_cartesian,
    calculate_relative_angle,
    cartesian_motion_to_angular,
)
from vehicles.entity.base_object import BaseObject


def shift(entity: BaseObject, displacement: np.ndarray) -> None:
    """Move an entity and its facing point together."""
    entity.position = (entity.position + displacement) #. .astype(entity.position.dtype)
    entity.facing_point = (entity.facing_point + displacement) # .astype(entity.facing_point.dtype)


def set_velocity(entity: BaseObject, velocity: np.ndarray) -> None:
    """Point an entity along a heading & speed """
    heading, speed = cartesian_motion_to_angular(velocity)
    entity.turn(calculate_relative_angle(entity.heading, heading))
    entity.velocity = float(np.clip(speed, -(entity.max_speed / 2), entity.max_speed))


def resolve_contact(second: BaseObject,
                    normal: np.ndarray,
                    overlap: float,
                    elasticity: float,
                    first: Optional[BaseObject] = None) -> None:
    """
    Separate two touching bodies and exchange velocity along the contact normal.

    Parameters
    ----------
    second : body pushed along `normal`
    normal : unit vector pointing from `first` toward `second`
    overlap : penetration depth
    elasticity : float for rebounding force - 0 = inelastic to 1 = perfectly elastic
    first : the other body, or None for immovable terrain
    """
    inverse_first = 0.0 if first is None else 1.0 / first.mass
    inverse_second = 1.0 / second.mass
    total = inverse_first + inverse_second

    if overlap > 0:
        if first is not None:
            shift(first, -normal * (inverse_first / total) * overlap)
        shift(second, normal * (inverse_second / total) * overlap)

    velocity_first = np.zeros(2) if first is None else angular_motion_to_cartesian(first.heading, first.velocity)
    velocity_second = angular_motion_to_cartesian(second.heading, second.velocity)

    closing = np.dot(velocity_second - velocity_first, normal)
    if closing >= 0:
        return

    impulse = -(1.0 + elasticity) * closing / total * normal
    if first is not None:
        set_velocity(first, velocity_first - impulse * inverse_first)
    set_velocity(second, velocity_second + impulse * inverse_second)
