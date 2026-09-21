"""
End-to-end validation: builds the full demo world and runs it live through
Renderer. Moved out of renderer.py so the renderer module stays pure
(Renderer class only, no demo/setup code).
"""
import pygame
from vehicles.world.renderer import Renderer

if __name__ == "__main__":
    from vehicles.world.controller import InputController
    from vehicles.world.world import World
    from vehicles.world.state import WorldState, EntityState
    from vehicles.entity.vehicle import Vehicle
    from vehicles.entity.base_object import BaseObject
    from vehicles.entity.angles import Angle, AngularType
    from vehicles.entity.senses import Sense, SensorType, SensorShape
    from vehicles.entity.behaviors.instinct import (
        Instinct, gate_strength, evade, approach, orbit, arrive, pursuit, separation
    )
    from vehicles.entity.species import Fish, Predator, Herbivore
    from vehicles.entity.environment import Rock, Tree, Grass

    record = []

    input_controller = InputController()
    renderer = Renderer(width=900, height=900, verbose=True)
    world = World(controller=input_controller, verbose=False)

    sight = Sense(type=SensorType.SIGHT,
                  shape=SensorShape.CONE,
                  range=200,
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

    # Second instinct: approach + orbit stacked together, so their outputs sum
    # each tick -- close in on whatever's strongest, then circle it once near.
    orbit_approach_instinct = Instinct()
    orbit_approach_instinct.add("approach", condition=gate_strength(0.01), action=approach)
    orbit_approach_instinct.add("orbit", condition=gate_strength(0.01), action=orbit)

    orbiter = Vehicle(
        mass=2,
        position=(350, 350),
        size=(10, 10),
        facing_point=(360, 360),
        sense=hearing,
        is_controlled=False,
        instinct=orbit_approach_instinct
    )

    other_obj = BaseObject(
        mass=10,
        position=(400, 400),
        size=(20, 12),
        facing_point=(400, 401),
    )

    # Arrive: seeks like approach, but tapers its throttle down as it closes
    # in on a nearby landmark instead of flooring it -- watch it settle near
    # the landmark rather than plowing into it or circling like orbiter does.
    arrive_instinct = Instinct()
    arrive_instinct.add("arrive", condition=gate_strength(0.01), action=lambda p, t: arrive(p, t, slow_radius=120.0))

    landmark = BaseObject(
        mass=10,
        position=(150, 450),
        size=(14, 14),
        facing_point=(150, 451),
    )

    arriver = Vehicle(
        mass=2,
        position=(150, 520),  # dist=70 to landmark -- comfortably inside
        size=(10, 10),         # hearing's range=100, so it won't flicker
        facing_point=(160, 520),  # in and out of detection near the boundary
        sense=hearing,
        is_controlled=False,
        instinct=arrive_instinct
    )

    # Pursuit: seeks where its target is HEADING, not where it currently is.
    # Give it a wider sense than the others so it can pick up a distant
    # moving target (orbiter, already circling other_obj/controlled_vehicle)
    # regardless of exactly where that circling has drifted to.
    pursuit_instinct = Instinct()
    pursuit_instinct.add("pursuit", condition=gate_strength(0.01), action=lambda p, t: pursuit(p, t, lookahead=3.0))

    wide_hearing = Sense(type=SensorType.HEARING,
                          shape=SensorShape.OMNI,
                          range=300,
                          field_of_view=Angle(type=AngularType.RADIANS, value=6.28),
                          noise=0.12,
                          xray=True,
                          falloff_exponent=1
                          )

    pursuer = Vehicle(
        mass=2,
        position=(100, 100),
        size=(10, 10),
        facing_point=(110, 100),
        sense=wide_hearing,
        max_speed=6,
        is_controlled=False,
        instinct=pursuit_instinct
    )

    separation_instinct = Instinct()
    separation_instinct.add("separation", condition=gate_strength(0.01), action=lambda p, t: separation(p, t, strength=0.6))

    personal_space_sense = Sense(type=SensorType.HEARING,
                                  shape=SensorShape.OMNI,
                                  range=60,
                                  field_of_view=Angle(type=AngularType.RADIANS, value=6.28),
                                  noise=0.05,
                                  xray=True,
                                  falloff_exponent=1
                                  )

    spacer_a = Vehicle(mass=2, position=(650, 150), size=(10, 10), facing_point=(651, 150),
                        sense=personal_space_sense, is_controlled=False, instinct=separation_instinct)
    spacer_b = Vehicle(mass=2, position=(665, 160), size=(10, 10), facing_point=(666, 160),
                        sense=personal_space_sense, is_controlled=False, instinct=separation_instinct)
    spacer_c = Vehicle(mass=2, position=(640, 170), size=(10, 10), facing_point=(641, 170),
                        sense=personal_space_sense, is_controlled=False, instinct=separation_instinct)

    # Fish: full Reynolds flocking (separation + alignment + cohesion) via
    # entity.species.Fish, pre-wired with same_type(Fish) so the school only
    # reacts to itself -- everything above is invisible to it for schooling
    # purposes, same as it's invisible to them.
    school = [
        Fish(position=(5 + (i % 3) * 20, 80 + (i // 3) * 20),
             facing_point=(5 + (i % 3) * 20 + 1, 80 + (i // 3) * 20))
        for i in range(20)
    ]

    # Predator + Herbivore herd: the predator (not_type(Predator)) hunts
    # every herbivore in range via pursuit, and each herbivore (same_type
    # (Predator)) evades it the moment it's sensed, while also keeping a
    # separation buffer (avoid) from its herd-mates and everything else.
    predator = Predator(
        position=(700, 700),
        facing_point=(700, 710),
    )

    herd = [
        Herbivore(position=(550 + (i % 3) * 40, 560 + (i // 3) * 40),
                  facing_point=(550 + (i % 3) * 40 + 1, 560 + (i // 3) * 40))
        for i in range(6)
    ]

    # Environment: a Rock barrier and a small Tree/Grass patch, off to the
    # side of the school/herd/predator cluster so they don't spawn tangled
    # up in each other.
    rock = Rock(
        position=(250, 650),
        facing_point=(250, 660),
    )

    trees = [
        Tree(
            position=(200, 200 + i * 100),
            facing_point=(201, 201 + i * 100))
        for i in range(2)
    ]

    grass_patch = [
        Grass(position=(600 + i * 20, 601),
              facing_point=(601 + i * 20 + 1, 601))
        for i in range(5)
    ]

    world.add_entity(controlled_vehicle)
    # world.add_entity([evader_a, evader_b, evader_c, evader_d, orbiter])
    # world.add_entity(other_obj)
    # world.add_entity([arriver, landmark])
    # world.add_entity(pursuer)
    # world.add_entity([spacer_a, spacer_b, spacer_c])
    world.add_entity(school)
    world.add_entity(predator)
    world.add_entity(herd)
    world.add_entity(rock)
    world.add_entity(trees)
    world.add_entity(grass_patch)

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
