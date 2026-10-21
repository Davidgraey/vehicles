"""
The simulation model

World is the headless heart of the environment: it owns the entities, the world bounds, the tick counter and an RNG,
and it advances the simulation one step at a time

It imports only numpy and our package contents. no pygame, no display
so it can be constructed and stepped in a headless process (tests, batch runs, neural-net training) where opening a
render window would be pointless or impossible.
"""
import copy
from typing import List, Optional

import numpy as np

from vehicles.entity.base_object import BaseObject
from vehicles.entity.environment import Plant
from vehicles.entity.species import Predator, Herbivore
from vehicles.entity.vehicle import Vehicle
from vehicles.world.collisions import resolve_contact
from vehicles.world.state import EntityState, TerrainState, WorldState
from vehicles.world.terrain import Terrain
from vehicles.world.world_view import Frame, build_observations, sense_world


class World:
    """Headless, steppable world"""

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
        self.terrain: List[Terrain] = []
        self.frame: Optional[Frame] = None

        self.controller = controller
        self.verbose = verbose

        # Collision bounciness: 0 = perfectly inelastic, 1.0 = perfect reflection
        self.restitution: float = 0.4

        # Entities added before the first reset() form the initial population; reset() restores exactly this set
        self._initial_entities: List[BaseObject] = []

    # ------------------------------ population ------------------------------
    def add_entity(self, entity: BaseObject | list[BaseObject]) -> None:
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

    def add_terrain(self, terrain: Terrain | list[Terrain]) -> None:
        """Add static terrain. Terrain persists across reset()."""
        if isinstance(terrain, list):
            self.terrain.extend(terrain)
        else:
            self.terrain.append(terrain)

    # ------------------------------ actions ------------------------------
    def sense(self) -> None:
        """Measure and sense the whole world; the result is kept as `frame`."""
        self.frame = sense_world(self.entities, self.terrain, self.tick)

    def observe(self) -> None:
        """Give each vehicle its observation from the latest `frame`."""
        for vehicle, observation in build_observations(self.entities, self.terrain, self.frame):
            vehicle.observe(observation)

    def act(self, commands: dict) -> None:
        """Apply motor commands, grow, and move every entity."""
        for entity in self.entities:
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
        self.frame = None
        self.entities = [copy.deepcopy(v) for v in self._initial_entities]
        return self.snapshot()

    def step(self) -> WorldState:
        """ key stepping """
        if (self.controller is not None):
            self.controller.process_events()
            commands = self.controller.get_commands()
        else:
            commands = {'turn': 0.0, 'accelerate': 0.0}

        self.sense()
        self.observe()
        self.act(commands)

        self.resolve_terrain()
        self.apply_bounds()
        self.check_collisions()

        self.tick += 1

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
                has_detections=len(v.detected_objects) > 0 if isinstance(v, Vehicle) else False,
                is_colliding=len(v.colliding) > 0,
                color=v.color,
                body_poly=v._get_body_render(),
                hunger=getattr(v, 'hunger', None),
                max_hunger=getattr(v, 'max_hunger', None),
                food=getattr(v, 'food', None),
                max_food=getattr(v, 'max_food', None),
                health=v.health,
                max_health=v.max_health,
                active_behavior=(", ".join(v.instinct.active_behaviors)
                                 if isinstance(v, Vehicle) and not v.is_controlled and v.instinct is not None
                                 else None),
                is_controlled=getattr(v, 'is_controlled', False)
            )
            for v in self.entities
        )
        return WorldState(
            tick=self.tick,
            width=self.width,
            height=self.height,
            entities=entities,
            terrain=tuple(
                TerrainState(
                    color=t.color,
                    points=tuple((float(x), float(y)) for x, y in t.shape.points),
                    thickness=t.shape.thickness,
                    closed=t.shape.closed,
                )
                for t in self.terrain
            ),
        )

    # ------------------------------ physics ------------------------------

    def resolve_terrain(self) -> None:
        """Push entities out of blocking terrain and bounce them off it."""
        barriers = [t for t in self.terrain if t.blocks_movement]

        for entity in self.entities:
            for barrier in barriers:
                position = entity.position.astype(np.float64)
                signed, normals = barrier.shape.contact(position[None, :])
                depth = entity.radius - signed[0]
                if depth > 0:
                    resolve_contact(entity, normals[0], depth, self.restitution)

    def apply_bounds(self) -> None:
        """clip positions"""
        for v in self.entities:
            v.position = np.clip(v.position, a_min=0, a_max=self.width)
            v.facing_point = np.clip(v.facing_point, a_min=0, a_max=self.width + 20)

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

        self.resolve_contacts(touching)

        return touching

    def resolve_contacts(self, touching: np.ndarray, restitution: Optional[float] = None) -> None:
        """
        Resolve every touching pair: separate and bounce them (see `resolve_contact`),
        then feed: a Vehicle touching a Plant eats from it; a Predator touching a
        Herbivore attacks it.

        `restitution` defaults to `self.restitution` when not given.
        """
        if restitution is None:
            restitution = self.restitution

        for i, j in np.argwhere(np.triu(touching, k=1)):
            a, b = self.entities[i], self.entities[j]

            delta = b.position.astype(np.float64) - a.position.astype(np.float64)
            distance = np.linalg.norm(delta)

            if distance > 1e-6:
                normal = delta / distance
            else:
                normal = np.array([1.0, 0.0])
                distance = 0.0

            resolve_contact(b, normal, (a.radius + b.radius) - distance, restitution, first=a)

            if isinstance(a, Vehicle) and isinstance(b, Plant):
                a.eat(b)
            elif isinstance(b, Vehicle) and isinstance(a, Plant):
                b.eat(a)
            elif isinstance(a, Predator) and isinstance(b, Herbivore):
                a.eat_prey(b)
            elif isinstance(b, Predator) and isinstance(a, Herbivore):
                b.eat_prey(a)
