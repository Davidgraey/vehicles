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
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from .state import EntityState, WorldState
from .vehicle import Vehicle


class World:
    """Headless, steppable Braitenberg-vehicle environment."""

    def __init__(self, width: int = 800, height: int = 600, seed: Optional[int] = None):
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
        self.entities: List[Vehicle] = []

        # Entities added before the first reset() form the initial population;
        # reset() restores exactly this set so runs are repeatable.
        self._initial_entities: List[Vehicle] = []

    # ------------------------------ population ------------------------------

    def add_entity(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the world and register it as part of the initial state."""
        self.entities.append(vehicle)
        self._initial_entities.append(copy.deepcopy(vehicle))

    # ------------------------------ gym-like API ------------------------------

    def reset(self) -> WorldState:
        """Restore tick and the initial entity population; return the opening snapshot."""
        self.tick = 0
        self.entities = [copy.deepcopy(v) for v in self._initial_entities]
        return self.snapshot()

    def step(self, dt: float = 1.0) -> WorldState:
        """Advance the simulation by ``dt`` and return the resulting snapshot.

        Each entity drives itself (Braitenberg-style) inside its own ``step``;
        the World then enforces bounds and advances the tick counter.
        """
        for vehicle in self.entities:
            vehicle.step(dt, self)
        self._apply_bounds()
        self.tick += 1
        return self.snapshot()

    def snapshot(self) -> WorldState:
        """Build an immutable, pygame-free view of the current state."""
        entities = tuple(
            EntityState(
                id=v.id,
                position=(float(v.position[0]), float(v.position[1])),
                heading=float(v.heading),
                size=tuple(v.size),
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
        """Wrap entity positions toroidally so they never leave the world."""
        for v in self.entities:
            v.position[0] %= self.width
            v.position[1] %= self.height
