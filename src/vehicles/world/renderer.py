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
        polygon_points = [self._translate_to_pygame_coords(int(x), int(y)) for x, y in vertices]

        color = pygame.Color(255, 0, 0, 32) if entity.has_detections else pygame.Color(0, 255, 100, 32)
        pygame.draw.polygon(self.screen, color, polygon_points, 2)  # width=2 for outline

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
    from vehicles.world.controller import InputController
    from vehicles.world.world import World
    from vehicles.world.state import WorldState, EntityState
    from vehicles.entity.vehicle import Vehicle
    from vehicles.entity.base_object import BaseObject
    from vehicles.entity.angles import Angle, AngularType
    from vehicles.entity.senses import Sense, SensorType, SensorShape
    from vehicles.entity.behaviors.instinct import Instinct, gate_strength, evade

    record = []

    input_controller = InputController()
    renderer = Renderer(width=800, height=800)
    world = World(controller=input_controller)

    sight = Sense(type=SensorType.SIGHT,
                  shape=SensorShape.CONE,
                  range=150,
                  field_of_view=Angle(type=AngularType.RADIANS, value=1.0),
                  noise=0.12,
                  xray=False,
                  falloff_exponent=1
                  )

    hearing = Sense(type=SensorType.HEARING,
                  shape=SensorShape.OMNI,
                  range=100,
                  field_of_view=Angle(type=AngularType.RADIANS, value=6.28),
                  noise=0.12,
                  xray=True,
                  falloff_exponent=1
                  )

    controlled_vehicle = Vehicle(
        mass=2,
        position=(300, 300),
        size=(10, 10),
        facing_point=(310, 310),
        sense=sight,
        is_controlled=True
    )

    evade_instinct = Instinct()
    evade_instinct.add("evade", condition=gate_strength(0.01), action=evade)

    evader_a = Vehicle(
        mass=2,
        position=(500, 300),
        size=(10, 10),
        facing_point=(510, 300),
        sense=hearing,
        is_controlled=False,
        instinct=evade_instinct
    )

    evader_b = Vehicle(
        mass=2,
        position=(400, 500),
        size=(10, 10),
        facing_point=(410, 500),
        sense=sight,
        is_controlled=False,
        instinct=evade_instinct
    )

    evader_c = Vehicle(
        mass=10,
        position=(600, 5250),
        size=(15, 15),
        facing_point=(410, 500),
        sense=hearing,
        is_controlled=False,
        instinct=evade_instinct
    )

    evader_d = Vehicle(
        mass=10,
        position=(695, 588),
        size=(15, 15),
        facing_point=(410, 500),
        sense=sight,
        is_controlled=False,
        instinct=evade_instinct
    )

    other_obj = BaseObject(
        mass=10,
        position=(400, 400),
        size=(20, 12),
        facing_point=(400, 401),
    )

    evader_a.max_speed, evader_b.max_speed, evader_c.max_speed, evader_d.max_speed = (5, 5, 5, 5)

    world.add_entity(controlled_vehicle)
    world.add_entity([evader_a, evader_b, evader_c, evader_d])
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

        # print("our vehicle: ", controlled_vehicle.position)

        # 2. Visualize (Renderer knows nothing about physics)
        renderer.render(state)

        # 3. Cleanup and tick our clock
        clock.tick(30)
        record.append(state)

    renderer.close()
    print(len(record))
