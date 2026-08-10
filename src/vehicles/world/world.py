"""
The simulation model.

`World` is the headless heart of the environment: it owns the entities, the world
bounds, the tick counter and an RNG, and it advances the simulation one step at a
time. It imports only numpy -- no pygame, no display -- so it can be constructed
and stepped in a headless process (tests, batch runs, neural-net training) where
opening a window would be pointless or impossible.

Rendering is deliberately somebody else's job: `World.step` returns an immutable
`WorldState` snapshot, and the `Renderer` turns that into pixels. The two never
share mutable state.

Usage (gym-like)::

    world = World(800, 600, seed=1)
    world.reset()
    while True:
        state = world.step(dt=1.0)   # autonomous; no action needed
        ...                          # render(state), log(state), etc.
"""
import copy
from typing import List, Optional
import numpy as np

from src.vehicles.world.state import EntityState, WorldState
from src.vehicles.entity.angles import Angle, AngularType
from src.vehicles.entity.base_object import BaseObject
from src.vehicles.entity.vehicle import Vehicle


class World:
    """Headless, steppable Braitenberg-vehicle environment."""

    def __init__(self, width: int = 800, height: int = 600, controller: Optional = None, seed: Optional[int] = None):
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

        # Entities added before the first reset() form the initial population;
        # reset() restores exactly this set so runs are repeatable.
        self._initial_entities: List[BaseObject] = []

    # ------------------------------ population ------------------------------
    def add_entity(self, entity: BaseObject) -> None:
        """Add a vehicle to the world and register it as part of the initial state."""
        self.entities.append(entity)
        self._initial_entities.append(copy.deepcopy(entity))

    # ------------------------------ actions ------------------------------
    def apply_commands(self, commands: dict):
        for entity in self.entities:
            if hasattr(entity, 'is_controlled'):
                print("entity commands: ", commands)
                _turn = Angle(AngularType.RADIANS, commands.get("turn", 0))
                # pygame moves in CCW?
                entity.turn(_turn)
                entity.accelerate(commands.get("accelerate", 0))

            entity.move()

            print(self.entities)

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
        self._apply_bounds()

        return self.snapshot()

    def snapshot(self) -> WorldState:
        """Build an immutable, pygame-free view of the current state."""
        entities = tuple(
            EntityState(
                id=v.id,
                position=(float(v.position[0]), float(v.position[1])),
                facing_point=(float(v.facing_point[0]), float(v.facing_point[1])),
                size=tuple(v.size),
                sense_poly=v._get_sensor_render() if isinstance(v, Vehicle) else None
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

    def _apply_bounds(self) -> None:
        """clip positions"""
        for v in self.entities:
            v.position = np.clip(v.position, a_min=0, a_max = self.width)
            v.facing_point = np.clip(v.facing_point, a_min=0, a_max = self.width+20)

