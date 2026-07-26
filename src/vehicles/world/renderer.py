"""
The simulation view.

`Renderer` owns the pygame window and turns an immutable `WorldState` snapshot
into pixels. It is a *pure view*: `render` reads a snapshot and draws it, and
never mutates the World. All pygame/display dependencies live here, so the model
(`World`) stays headless and importable without a display.
"""
import math
import pygame as pg
from .state import WorldState


class Renderer:
    """Draws `WorldState` snapshots to a pygame window."""

    def __init__(
        self,
        width: int,
        height: int,
        caption: str = "Braitenberg Vehicles",
        background: str = "floralwhite",
    ):
        pg.init()
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption(caption)
        self.background = pg.Color(background)

    def render(self, state: WorldState) -> None:
        """Clear the screen, draw every entity, then present the frame."""
        self.screen.fill(self.background)
        for entity in state.entities:
            self._draw_entity(entity)
        pg.display.flip()

    def _draw_entity(self, entity) -> None:
        """Draw one entity as a body circle with a line marking its facing."""
        x, y = entity.position
        radius = max(entity.size) // 2

        pg.draw.circle(self.screen, pg.Color("steelblue"), (int(x), int(y)), radius)

        # Facing indicator: heading 0 == +x, counter-clockwise positive.
        tip = (
            int(x + math.cos(entity.heading) * radius),
            int(y + math.sin(entity.heading) * radius),
        )
        pg.draw.line(self.screen, pg.Color("black"), (int(x), int(y)), tip, 2)

    def close(self) -> None:
        """Tear down the pygame window."""
        pg.quit()
