"""
Runnable entry point for the windowed simulation.

`SimulationApp` wires the headless `World` (model) to the `Renderer` (view) and
owns everything that only matters when a human is watching: the pygame clock, the
input handling, and the real-time loop. The loop itself is the gym-like triple --
``process_input -> world.step -> renderer.render`` -- so the model can also be
driven step-by-step from a headless script without this class at all.

Run it with::

    python -m vehicles.world.pygame_main
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pygame as pg

from .renderer import Renderer
from .vehicle import SimpleVehicle
from .world import World


class SimulationApp:
    """Real-time driver: steps the World and renders it at a fixed tick rate."""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        tick_rate: int = 60,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.tick_rate = tick_rate
        self.dt = dt
        self.running = True

        self.world = World(width, height, seed=seed)
        self._populate(width, height)

        self.renderer = Renderer(width, height)
        self.clock = pg.time.Clock()

    def _populate(self, width: int, height: int) -> None:
        """Seed the world with a few autonomous vehicles heading in random directions."""
        for i in range(3):
            self.world.add_entity(
                SimpleVehicle(
                    position=(width * (i + 1) / 4.0, height / 2.0),
                    heading=float(self.world.rng.uniform(0, 2 * np.pi)),
                    id=i,
                )
            )

    def process_input(self) -> None:
        """Handle window/quit events. The simulation itself is autonomous."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False

    def run(self) -> None:
        """Reset and run the real-time loop until the user quits."""
        self.world.reset()
        while self.running:
            self.process_input()
            state = self.world.step(self.dt)
            self.renderer.render(state)
            self.clock.tick(self.tick_rate)
        self.renderer.close()


def main() -> None:
    SimulationApp().run()


if __name__ == "__main__":
    main()
