import numpy as np
from typing import Optional

from vehicles.entity.vehicle import Vehicle
from vehicles.entity.base_object import BaseObject
from vehicles.entity.angles import Angle, AngularType
from vehicles.entity.senses import Sense, SensorType, SensorShape
from vehicles.entity.behaviors.instinct import Instinct, same_type, hungry_for, separation, flock, pursuit, evade, arrive
from vehicles.entity.environment import Plant


class Fish(Vehicle):
    """
    Schools with other Fish, and only other Fish: separation keeps a
    little personal space. Once hungry enough, also seeks out Plant
    (find_food, same hungry_for()/arrive() pattern as Herbivore): it eases
    off near the plant instead of flying past, so it stays close long enough
    to actually eat.
    """
    def __init__(self,
                 position: tuple[int, int],
                 facing_point: tuple[int, int],
                 mass: int = 2,
                 size: tuple[int, int] = (8, 8),
                 personal_space: float = 25.0,
                 school_range: float = 120.0,
                 hunger_threshold: float = 20.0,
                 max_speed: Optional[float] = 5,
                 speed: Optional[float] = 0.05,
                 is_controlled: bool = False):

        sense = Sense(
            type=SensorType.ELECTROMAG,
            shape=SensorShape.OMNI,
            range=school_range,
            field_of_view=Angle(type=AngularType.RADIANS, value=6.28319),
            noise=0.05,
            xray=False,
            falloff_exponent=1,
        )

        instinct = Instinct()
        # tight radius
        instinct.add(
            "separation",
            condition=same_type(Fish, max_distance=personal_space),
            action=lambda p, t: separation(p, t, strength=0.6),
        )
        # full sense range
        instinct.add(
            "flock",
            condition=same_type(Fish),
            action=lambda p, t: flock(p, t, speed=0.5),
        )
        # once hungry enough, go find a Plant and ease in on it
        instinct.add(
            "find_food",
            condition=hungry_for(Plant, hunger_threshold=hunger_threshold),
            action=lambda p, t: arrive(p, t, slow_radius=40.0),
        )

        super().__init__(
            mass=mass,
            position=position,
            size=size,
            facing_point=facing_point,
            sense=sense,
            metabolism=0.05,
            eat_rate=0.2,
            max_speed=max_speed,
            speed=speed,
            is_controlled=is_controlled,
            instinct=instinct,
        )
        self.school_range = school_range
        self.personal_space = personal_space

        # Appearance -- silver body, drawn as a fish silhouette instead of
        # the generic circle (see _get_body_render below)
        self.color = "silver"

    def _get_body_render(self) -> np.ndarray:
        """
        A small elongated body with a forked tail instead of the generic
        circle every other entity draws by default. Built the same way
        Vehicle._get_sensor_render() builds the sense cone: local offsets
        (forward = +y) rotated by heading using the renderer's y-flip
        convention, then translated to world position.
        """
        length = float(np.max(self.size)) * 0.9
        width = float(np.max(self.size)) * 0.5

        # local silhouette, forward = +y, nose first, wound consistently
        # for polygon fill: nose -> right shoulder -> right tail-fin tip ->
        # tail notch -> left tail-fin tip -> left shoulder
        local = np.array([
            [0.0,           length],
            [width * 0.5,   length * 0.35],
            [width,        -length * 0.55],
            [0.0,          -length * 0.25],
            [-width,       -length * 0.55],
            [-width * 0.5,  length * 0.35],
        ])

        self.heading.cast_as_radians()
        heading_rad = -self.heading.value  # same renderer y-flip convention as _get_sensor_render

        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        world_rel_x = local[:, 0] * cos_h - local[:, 1] * sin_h
        world_rel_y = local[:, 0] * sin_h + local[:, 1] * cos_h

        return np.column_stack([
            self.position[0] + world_rel_x,
            self.position[1] + world_rel_y,
        ])


class Predator(Vehicle):
    """
    Hunts Herbivores, and only Herbivores. Uses pursuit -- steering toward
    where its strongest detection is HEADING, not just where it currently
    is -- so it leads a moving target instead of always trailing it.
    Scoped with same_type(Herbivore), same pattern Fish uses for schooling
    -- Fish, other Predators, and scenery are all left alone. Once hungry
    enough, also falls back to find_food (hungry_for(Plant)/arrive()) when
    there's no prey to pursue.
    """

    def __init__(self,
                 position: tuple[int, int],
                 facing_point: tuple[int, int],
                 mass: int = 3,
                 size: tuple[int, int] = (16, 16),
                 lookahead: float = 3.0,
                 hunger_threshold: float = 20.0,
                 max_speed: Optional[float] = 6,
                 speed: Optional[float] = 0.2,
                 is_controlled: bool = False):

        sense = Sense(
            type=SensorType.SIGHT,
            shape=SensorShape.CONE,
            range=200,
            field_of_view=Angle(type=AngularType.RADIANS, value=2 * np.pi),
            noise=0.05,
            xray=False,
            falloff_exponent=1.1,
        )

        instinct = Instinct()
        instinct.add(
            "pursuit",
            condition=same_type(Herbivore),
            action=lambda p, t: pursuit(p, t, lookahead=lookahead),
        )
        # once hungry enough, go find a Plant and ease in on it
        instinct.add(
            "find_food",
            condition=hungry_for(Plant, hunger_threshold=hunger_threshold),
            action=lambda p, t: arrive(p, t, slow_radius=40.0),
        )

        super().__init__(
            mass=mass,
            position=position,
            size=size,
            facing_point=facing_point,
            sense=sense,
            max_speed=max_speed,
            speed=speed,
            is_controlled=is_controlled,
            instinct=instinct,
        )

        # Appearance -- larger, darker, sharper-nosed than Fish, so a
        # predator reads as a predator at a glance
        self.color = "firebrick"

    def _get_body_render(self) -> np.ndarray:
        """
        A bigger, sharper wedge than Fish's rounder forked-tail silhouette.
        Built the same way -- local offsets (forward = +y) rotated by
        heading using the renderer's y-flip convention, then translated to
        world position.
        """
        length = float(np.max(self.size)) * 1.0
        width = float(np.max(self.size)) * 0.55

        # local silhouette, forward = +y: nose -> wide forward jaw ->
        # narrower tail -> tail notch -> narrower tail -> wide forward jaw
        local = np.array([
            [0.0,            length],
            [width,          length * -0.2],
            [width * 0.4,    length * -0.7],
            [0.0,            length * -0.45],
            [-width * 0.4,   length * -0.7],
            [-width,         length * -0.2],
        ])

        self.heading.cast_as_radians()
        heading_rad = -self.heading.value  # same renderer y-flip convention as _get_sensor_render

        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        world_rel_x = local[:, 0] * cos_h - local[:, 1] * sin_h
        world_rel_y = local[:, 0] * sin_h + local[:, 1] * cos_h

        return np.column_stack([
            self.position[0] + world_rel_x,
            self.position[1] + world_rel_y,
        ])


class Herbivore(Vehicle):
    """
    Skittish prey: evades Predators outright, and separately keeps a
    general "avoid" buffer from everything else too -- other herbivores,
    scenery, whatever's nearby -- so it doesn't blunder into things while
    it's not busy fleeing. Two behaviors, same_type()-scoped like Fish and
    Predator:

      evade  -- same_type(Predator): the moment a Predator is detected,
                turn away and reverse hard. This alone would let it wander
                blindly into everything else, hence:
      avoid  -- same_type(BaseObject, max_distance=personal_space), minus
                any Plant: matches literally anything close by (reusing
                same_type() as a type-agnostic "within range" filter)
                except food, so it doesn't fight find_food by keeping the
                one thing it's trying to reach at arm's length.
      find_food -- hungry_for(Plant): once hungry enough, seek out any
                detected Plant via arrive() -- eases off near it instead of
                flying past, so it stays close enough to eat. Eating itself
                happens on contact, in World.resolve_contacts().
    """

    def __init__(self,
                 position: tuple[int, int],
                 facing_point: tuple[int, int],
                 mass: int = 2,
                 size: tuple[int, int] = (10, 10),
                 personal_space: float = 30.0,
                 hunger_threshold: float = 20.0,
                 max_speed: Optional[float] = 8,
                 speed: Optional[float] = 0.1,
                 is_controlled: bool = False):

        sense = Sense(
            type=SensorType.HEARING,
            shape=SensorShape.OMNI,
            range=150,
            field_of_view=Angle(type=AngularType.RADIANS, value=2 * np.pi),
            noise=0.05,
            xray=False,
            falloff_exponent=1,
        )

        instinct = Instinct()
        # flee any detected Predator, full priority
        instinct.add(
            "evade",
            condition=same_type(Predator),
            action=evade,
        )
        # otherwise, don't crowd whatever else is nearby -- except food,
        # or this would fight find_food by keeping it out of eating range
        instinct.add(
            "avoid",
            condition=lambda p, t: [
                d for d in same_type(BaseObject, max_distance=personal_space)(p, t)
                if not isinstance(d.get('object'), Plant)
            ],
            action=lambda p, t: separation(p, t, strength=0.5),
        )
        # once hungry enough, go find a Plant and ease in on it
        instinct.add(
            "find_food",
            condition=hungry_for(Plant, hunger_threshold=hunger_threshold),
            action=lambda p, t: arrive(p, t, slow_radius=40.0),
        )

        super().__init__(
            mass=mass,
            position=position,
            size=size,
            facing_point=facing_point,
            sense=sense,
            max_speed=max_speed,
            speed=speed,
            is_controlled=is_controlled,
            instinct=instinct,
        )
        self.personal_space = personal_space

        # Appearance -- warm, earthy tone and a soft rounded body, distinct
        # from Fish's silver forked tail and Predator's sharp firebrick wedge
        self.color = "peru"

    def _get_body_render(self) -> np.ndarray:
        """
        A gentle rounded-oval body with a blunt tail -- no fin notch like
        Fish, no sharp jaw like Predator. Built the same way: local offsets
        (forward = +y) rotated by heading using the renderer's y-flip
        convention, then translated to world position.
        """
        length = float(np.max(self.size)) * 0.85
        width = float(np.max(self.size)) * 0.55

        local = np.array([
            [0.0,            length],
            [width,          length * 0.3],
            [width * 0.8,   -length * 0.5],
            [0.0,           -length * 0.7],
            [-width * 0.8,  -length * 0.5],
            [-width,         length * 0.3],
        ])

        self.heading.cast_as_radians()
        heading_rad = -self.heading.value  # same renderer y-flip convention as _get_sensor_render

        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        world_rel_x = local[:, 0] * cos_h - local[:, 1] * sin_h
        world_rel_y = local[:, 0] * sin_h + local[:, 1] * cos_h

        return np.column_stack([
            self.position[0] + world_rel_x,
            self.position[1] + world_rel_y,
        ])
