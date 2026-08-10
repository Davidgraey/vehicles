"""
The simulation view.
Pure visualizer: depends only on WorldState snapshots and pygame.
No knowledge of physics, sensors, or Braitenberg logic.
"""
import pygame
import numpy as np
from typing import Any, Protocol
from src.vehicles.world.state import WorldState


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

    def render(self, state: WorldState) -> None:
        """Clear the screen, draw every entity, then present the frame."""
        self.screen.fill(self.background)

        for entity in state.entities:
            self._draw_entity(entity)
            if entity.sense_poly is not None:
                self._draw_senses(entity)
                # self._draw_bounds(entity)

        pygame.display.flip()

    def _translate_to_pygame_coords(self, x_value, y_value):

        return x_value, self.height - y_value

    def _draw_entity(self, entity: DrawableEntity) -> None:
        """Draw one entity as a body circle with a line marking its facing."""
        x, y = self._translate_to_pygame_coords(entity.position[0],
                                                entity.position[1]
                                                )
        facing_x, facing_y = self._translate_to_pygame_coords(entity.facing_point[0],
                                                              entity.facing_point[1]
                                                              )

        # TODO; fix this entitysize is an x,y size.. needs refinement
        radius = max(np.max(entity.size) // 2, 2)

        # Draw object
        pygame.draw.circle(self.screen, pygame.Color("steelblue"), (int(x), int(y)), radius)

        # Draw facing indicator
        pygame.draw.line(self.screen,
                         pygame.Color("black"),
                         (int(x), int(y)),
                         (int(facing_x), int(facing_y)),
                         2)

    def _draw_senses(self, entity: DrawableEntity) -> None:
        """Draw one entity as a body circle with a line marking its facing."""
        vertices = entity.sense_poly
        # Pygame expects list of (x, y) tuples
        poly_points = [self._translate_to_pygame_coords(int(x), int(y)) for x, y in vertices]


        color = pygame.Color(0, 255, 100, 32)
        pygame.draw.polygon(self.screen, color, poly_points, 2)  # width=2 for outline

    def handle_events(self, events: list[pygame.event.Event]) -> bool:
        """Process pygame events. Returns False if QUIT is requested."""
        for event in events:
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()

if __name__ == "__main__":
    # Main loop example (shows strict separation)
    from src.vehicles.world.controller import InputController
    from src.vehicles.world.world import World
    from src.vehicles.world.state import WorldState, EntityState
    from src.vehicles.entity.vehicle import Vehicle
    from src.vehicles.entity.base_object import BaseObject
    from src.vehicles.entity.angles import Angle, AngularType
    from src.vehicles.entity.senses import Sense, SensorType, SensorShape

    record = []

    input_controller = InputController()
    renderer = Renderer(width=800, height=800)
    world = World(controller=input_controller)

    sight = Sense(type=SensorType.SIGHT,
                  shape=SensorShape.CONE,
                  range=15,
                  field_of_view=Angle(type=AngularType.RADIANS, value=1.0),
                  noise=0.12,
                  xray=False,
                  falloff_exponent=1.0
                  )

    controlled_vehicle  = Vehicle(
        mass=1.2,
        position=(300, 300),
        size=(10, 10),
        facing_point=(310, 310),
        sense=sight,
        is_controlled=True
    )

    other_obj = BaseObject(
        mass=10,
        position=(400, 400),
        size=(20.0, 12.0),
        facing_point=(400, 401),
    )

    world.add_entity(controlled_vehicle)
    world.add_entity(other_obj)

    running = True
    clock = pygame.time.Clock()

    while running:
        # events = pygame.event.get()
        # # print(events)
        # if not renderer.handle_events(events):
        #     running = False

        # quit = input_controller.process_events(events)
        # if quit:
        #     running = False

        # 1. Simulate (World knows nothing about pixels)
        state = world.step()

        print("our vehicle: ", controlled_vehicle.position)

        # 2. Visualize (Renderer knows nothing about physics)
        renderer.render(state)

        # 3. Cleanup and tick our clock
        clock.tick(30)
        record.append(state)


    renderer.close()
    print(len(record))
