"""
The simulation model.

`World` is the headless heart of the environment: it owns the entities, the world
bounds, the tick counter and an RNG, and it advances the simulation one step at a
time. It imports only numpy -- no pygame, no display -- so it can be constructed
and stepped in a headless process (tests, batch runs, neural-net training) where
opening a window would be pointless or impossible.

"""
import copy
from typing import List, Optional
import numpy as np

from vehicles.world.state import EntityState, WorldState
from vehicles.entity.base_object import BaseObject
from vehicles.entity.vehicle import Vehicle
from vehicles.entity.environment import Plant
from vehicles.entity.angles import calculate_relative_angle, angular_motion_to_cartesian, cartesian_motion_to_angular


class World:
    """Headless, steppable Braitenberg-vehicle environment."""

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 controller: Optional = None,
                 seed: Optional[int] = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        width : world width in units (== pixels when rendered)
        height : world height in units
        seed : optional RNG seed for reproducible runs
        """
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        self.tick = 0
        self.entities: List[BaseObject] = []

        self.controller = controller
        self.verbose = verbose

        # Collision bounciness: 0 = perfectly inelastic (colliding objects
        # leave the impact moving together along the normal, no bounce),
        # 1 = perfectly elastic (no energy lost). Below 1 so every impact
        # bleeds off some speed, matching "velocity reduces with impact."
        self.restitution: float = 0.4

        # Entities added before the first reset() form the initial population;
        # reset() restores exactly this set so runs are repeatable.
        self._initial_entities: List[BaseObject] = []

    # ------------------------------ population ------------------------------
    def add_entity(self, entity: BaseObject|list[BaseObject]) -> None:
        """Add a vehicle to the world and register it as part of the initial state."""
        if isinstance(entity, list):
            for _e in entity:
                _e.world_bounds = (self.width, self.height)
                self.entities.append(_e)
                self._initial_entities.append(copy.deepcopy(_e))
        else:
            entity.world_bounds = (self.width, self.height)
            self.entities.append(entity)
            self._initial_entities.append(copy.deepcopy(entity))

    # ------------------------------ actions ------------------------------
    def apply_commands(self, commands: dict):
        for entity in self.entities:
            if hasattr(entity, 'sense'):
                entity.perceive(self.entities)

            if hasattr(entity, 'apply_motor_commands'):
                entity.apply_motor_commands(
                    turn=commands.get("turn", 0.0),
                    accelerate=commands.get("accelerate", 0.0)
                )

            if hasattr(entity, 'grow'):
                entity.grow()

            entity.move()

    # ------------------------------ gym-like API ------------------------------
    def reset(self) -> WorldState:
        """Restore tick and the initial entity population; return the opening snapshot."""
        self.tick = 0
        self.entities = [copy.deepcopy(v) for v in self._initial_entities]
        return self.snapshot()

    def step(self) -> WorldState:
        """ key stepping """
        if (self.controller is not None):
            self.controller.process_events()
            commands = self.controller.get_commands()
        else:
            commands = {'turn': 0.0, 'accelerate': 0.0}

        # 2. Apply physics, update positions, etc. using `commands`
        self.apply_commands(commands)

        #3. apply bounds -- wrap the world
        self.apply_bounds()

        #4. who's touching whom
        self.check_collisions()

        return self.snapshot()

    def snapshot(self) -> WorldState:
        """Build an immutable, pygame-free view of the current state."""

        entities = tuple(
            EntityState(
                id=v.id,
                position=(float(v.position[0]), float(v.position[1])),
                facing_point=(float(v.facing_point[0]), float(v.facing_point[1])),
                size=tuple(v.size),
                sense_poly=v._get_sensor_render() if isinstance(v, Vehicle) else None,
                has_detections = len(v.detected_objects) > 0 if isinstance(v, Vehicle) else False,
                is_colliding = len(v.colliding) > 0,
                color = v.color,
                body_poly = v._get_body_render(),
                hunger = getattr(v, 'hunger', None),
                max_hunger = getattr(v, 'max_hunger', None),
                food = getattr(v, 'food', None),
                max_food = getattr(v, 'max_food', None),
                active_behavior = (", ".join(v.instinct.active_behaviors)
                                    if isinstance(v, Vehicle) and not v.is_controlled and v.instinct is not None
                                    else None),
                is_controlled = getattr(v, 'is_controlled', False)
            )
            for v in self.entities
        )
        return WorldState(
            tick=self.tick,
            width=self.width,
            height=self.height,
            entities=entities,
        )

    # ------------------------------ physics ------------------------------

    def apply_bounds(self) -> None:
        """clip positions"""
        for v in self.entities:
            v.position = np.clip(v.position, a_min=0, a_max = self.width)
            v.facing_point = np.clip(v.facing_point, a_min=0, a_max = self.width+20)

    def check_collisions(self) -> np.ndarray:
        """
        All-pairs circle-circle touch test over every entity, in one shot.

        The numpy shortcut: stack every position into an (N, 2) array and
        every radius into an (N,) array, then let broadcasting build the
        whole NxN distance matrix at once --
            deltas    = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
            distances = np.linalg.norm(deltas, axis=-1)                 # (N, N)
        instead of a Python double loop comparing every pair by hand. Two
        entities are touching when the distance between their centers is
        <= the sum of their radii.

        Also stashes each entity's current contacts onto `entity.colliding`
        (mirroring how senses stash onto `Vehicle.detected_objects`), and
        returns the raw NxN boolean matrix (diagonal forced False -- an
        entity doesn't collide with itself) for anyone who wants it directly.
        """
        n = len(self.entities)
        if n < 2:
            for v in self.entities:
                v.colliding = []
            return np.zeros((n, n), dtype=bool)

        # float64 on purpose: positions are stored float16 elsewhere, and
        # squaring a difference of more than ~250 units overflows float16.
        positions = np.array([e.position for e in self.entities], dtype=np.float64)
        radii = np.array([e.radius for e in self.entities], dtype=np.float64)

        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        touch_radius = radii[:, None] + radii[None, :]

        touching = distances <= touch_radius
        np.fill_diagonal(touching, False)

        for i, entity in enumerate(self.entities):
            entity.colliding = [self.entities[j] for j in np.where(touching[i])[0]]

        self.resolve_collisions(touching)
        self.resolve_feeding(touching)

        return touching

    def resolve_collisions(self, touching: np.ndarray, restitution: Optional[float] = None) -> None:
        """
        Physically-informed response to every touching pair: separate the
        overlapping bodies, then exchange velocity via a mass-weighted
        impulse along the collision normal (standard 2D circle-circle
        impulse resolution). Heading and speed both fall out of that impulse
        instead of being scripted separately -- an object is "redirected"
        because its post-collision velocity vector genuinely points a new
        way, and a collision "reduces velocity" because `restitution` < 1
        bleeds off some of the closing speed on every hit.

        Mass does the rest on its own: every push and every velocity change
        below is split by inverse mass (1/mass), so a heavy object (small
        1/mass) barely moves and barely changes direction, while a light one
        gets shoved and redirected hard -- exactly "heavy objects redirect
        less and are less affected by collisions," with no special-casing.

        `restitution` defaults to `self.restitution` (0..1, bounciness: 0 =
        perfectly inelastic, 1 = perfectly elastic) when not given directly.
        """
        if restitution is None:
            restitution = self.restitution

        for i, j in np.argwhere(np.triu(touching, k=1)):
            a, b = self.entities[i], self.entities[j]

            inv_mass_a = 1.0 / a.mass
            inv_mass_b = 1.0 / b.mass
            total_inv_mass = inv_mass_a + inv_mass_b

            pos_a = a.position.astype(np.float64)
            pos_b = b.position.astype(np.float64)

            delta = pos_b - pos_a
            distance = np.linalg.norm(delta)

            if distance > 1e-6:
                normal = delta / distance
            else:
                # centers exactly stacked -- there's no meaningful direction
                # to separate along, so pick one
                normal = np.array([1.0, 0.0])
                distance = 0.0

            # --- separate: push apart along the normal, split by mass ----
            overlap = (a.radius + b.radius) - distance

            if overlap > 0:
                push_a = (inv_mass_a / total_inv_mass) * overlap
                push_b = (inv_mass_b / total_inv_mass) * overlap

                a.position = (pos_a - normal * push_a).astype(a.position.dtype)
                b.position = (pos_b + normal * push_b).astype(b.position.dtype)
                a.facing_point = (a.facing_point.astype(np.float64) - normal * push_a).astype(a.facing_point.dtype)
                b.facing_point = (b.facing_point.astype(np.float64) + normal * push_b).astype(b.facing_point.dtype)

            # --- redirect + slow: mass-weighted impulse along the normal --
            vel_a = angular_motion_to_cartesian(a.heading, a.velocity)
            vel_b = angular_motion_to_cartesian(b.heading, b.velocity)

            relative_velocity = vel_b - vel_a
            closing_speed = np.dot(relative_velocity, normal)

            if closing_speed < 0:
                # only apply an impulse when they're actually closing --
                # otherwise they're already separating and an impulse here
                # would inject energy instead of removing it
                impulse_mag = -(1.0 + restitution) * closing_speed / total_inv_mass
                impulse = impulse_mag * normal

                vel_a = vel_a - impulse * inv_mass_a
                vel_b = vel_b + impulse * inv_mass_b

                new_heading_a, new_speed_a = cartesian_motion_to_angular(vel_a)
                new_heading_b, new_speed_b = cartesian_motion_to_angular(vel_b)

                # turn() (not a direct heading assignment) so facing_point
                # rotates in lockstep with heading -- keeps the body polygon
                # and sense cone pointed the same way the renderer's facing
                # line is
                a.turn(calculate_relative_angle(a.heading, new_heading_a))
                b.turn(calculate_relative_angle(b.heading, new_heading_b))

                a.velocity = float(np.clip(new_speed_a, -(a.max_speed / 2), a.max_speed))
                b.velocity = float(np.clip(new_speed_b, -(b.max_speed / 2), b.max_speed))

    def resolve_feeding(self, touching: np.ndarray) -> None:
        """
        Feeding response to every touching pair: a Vehicle touching a
        Plant eats from it, separate from the physical collision response
        in `resolve_collisions`.
        """
        for i, j in np.argwhere(np.triu(touching, k=1)):
            a, b = self.entities[i], self.entities[j]

            if isinstance(a, Vehicle) and isinstance(b, Plant):
                a.eat(b)
            elif isinstance(b, Vehicle) and isinstance(a, Plant):
                b.eat(a)
