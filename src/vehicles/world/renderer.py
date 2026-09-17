"""
The simulation view.
Pure visualizer: depends only on WorldState snapshots and pygame.
No knowledge of physics, sensors, or Braitenberg logic.
"""
import pygame
import numpy as np
from typing import Any, Protocol
from vehicles.world.state import WorldState


class DrawableEntity(Protocol):
    """Minimal interface expected from any simulated entity."""
    def __getattr__(self, name: str) -> Any: ...


class Renderer:
    """Draws `WorldState` snapshots to a pygame window."""
    def __init__(
            self,
            width: int = 800,
            height: int = 800,
            caption: str = "Braitenberg Vehicles",
            background_color: str = "floralwhite",
    ):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.background = pygame.Color(background_color)
        self.font = pygame.font.SysFont(None, 14)

    def render(self, state: WorldState) -> None:
        """Clear the screen, draw every entity, then present the frame."""
        self.screen.fill(self.background)

        for entity in state.entities:
            self._draw_entity(entity)
            if entity.sense_poly is not None:
                self._draw_senses(entity)
                # self._draw_bounds(entity)
            if entity.hunger is not None:
                self._draw_stat_bar(entity, entity.hunger, 100, pygame.Color(255, 0, 0, 60))
            if entity.food is not None and entity.max_food is not None:
                self._draw_stat_bar(entity, entity.food, entity.max_food, pygame.Color(0, 200, 0, 60))
            if entity.active_behavior:
                self._draw_behavior_label(entity)

        pygame.display.flip()

    def _translate_to_pygame_coords(self, x_value, y_value):

        return x_value, self.height - y_value

    def _draw_entity(self, entity: DrawableEntity) -> None:
        """Draw one entity's body (its own silhouette if it has one, else a
        plain circle), with a line marking its facing."""
        x, y = self._translate_to_pygame_coords(entity.position[0],
                                                entity.position[1]
                                                )
        facing_x, facing_y = self._translate_to_pygame_coords(entity.facing_point[0],
                                                              entity.facing_point[1]
                                                              )

        # Colliding always wins, regardless of the entity's own color --
        # it's a transient alert, not part of its identity
        body_color = pygame.Color("orangered") if getattr(entity, "is_colliding", False) \
            else pygame.Color(getattr(entity, "color", "steelblue"))

        body_poly = getattr(entity, "body_poly", None)
        if body_poly is not None:
            polygon_points = [self._translate_to_pygame_coords(int(px), int(py)) for px, py in body_poly]
            pygame.draw.polygon(self.screen, body_color, polygon_points)
        else:
            # TODO; fix this entitysize is an x,y size.. needs refinement
            radius = max(np.max(entity.size) // 2, 2)
            pygame.draw.circle(self.screen, body_color, (int(x), int(y)), radius)

        # Draw facing indicator -- only for controllable vehicles
        if getattr(entity, "is_controlled", False):
            pygame.draw.line(self.screen,
                             pygame.Color("black"),
                             (int(x), int(y)),
                             (int(facing_x), int(facing_y)),
                             2)

    def _draw_senses(self, entity: DrawableEntity) -> None:
        """Draw one entity as a body circle with a line marking its facing."""
        vertices = entity.sense_poly
        # Pygame expects list of (x, y) tuples
        polygon_points = [self._translate_to_pygame_coords(int(x), int(y)) for x, y in vertices]

        color = pygame.Color(255, 0, 0, 32) if entity.has_detections else pygame.Color(0, 255, 100, 32)
        pygame.draw.polygon(self.screen, color, polygon_points, 2)  # width=2 for outline

    def _draw_stat_bar(self, entity: DrawableEntity, value: float, max_value: float, color: pygame.Color) -> None:
        """Draws a low-alpha fill bar just right of entity.position, sized by value/max_value."""
        if max_value <= 0:
            return

        x, y = self._translate_to_pygame_coords(entity.position[0], entity.position[1])
        radius = max(np.max(entity.size) // 2, 2)
        bar_width, bar_height = 16, 3

        fraction = np.clip(value / max_value, 0.0, 1.0)
        fill_rect = pygame.Rect(int(x + radius + 4), int(y - bar_height // 2), int(bar_width * fraction), bar_height)
        pygame.draw.rect(self.screen, color, fill_rect)

    def _draw_behavior_label(self, entity: DrawableEntity) -> None:
        """Renders the entity's active behavior name(s), centered just above its body."""
        x, y = self._translate_to_pygame_coords(entity.position[0], entity.position[1])
        radius = max(np.max(entity.size) // 2, 2)

        label = self.font.render(entity.active_behavior, True, pygame.Color("black"))
        label_x = int(x - label.get_width() / 2)
        label_y = int(y - radius - label.get_height() - 2)
        self.screen.blit(label, (label_x, label_y))

    def handle_events(self, events: list[pygame.event.Event]) -> bool:
        """Process pygame events. Returns False if QUIT is requested."""
        for event in events:
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()
