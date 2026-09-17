import numpy as np
from vehicles.entity.base_object import BaseObject
from vehicles.entity.angles import Angle, AngularType
from vehicles.entity.senses import SensorType, SensorShape, Sense
from vehicles.entity.behaviors.instinct import Instinct


class Vehicle(BaseObject):
    def __init__(self,
                 mass: int,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 facing_point: tuple[int, int],
                 sense:Sense,
                 is_controlled: bool = True,
                 instinct: Instinct = None
                 ):
        super().__init__(mass=mass, position=position, size=size, facing_point=facing_point)

        self.sense = sense
        self.is_controlled = is_controlled
        self.instinct = instinct  # Optional behavior system
        self.detected_objects = []  # List of detected object positions

    def perceive(self, instance_objects: list):
        """ Call sense.perceive and track detected objects """
        detected, strength = self.sense.perceive(self, instance_objects)

        other_objects = [obj for obj in instance_objects if obj != self]
        self.detected_objects = []

        if len(detected) > 0:
            for idx, is_detected in enumerate(detected):
                if is_detected and idx < len(other_objects):
                    self.detected_objects.append({
                        'position': tuple(other_objects[idx].position),
                        'strength': float(strength[idx])
                    })

        return detected, strength

    def apply_motor_commands(self, turn: float = 0.0, accelerate: float = 0.0) -> None:
        """
        Apply motor commands. Source depends on control mode:
          - controlled: use the provided (turn, accelerate) from the controller
          - autonomous: ignore controller input entirely; use instinct if available
        """
        if not self.is_controlled:
            turn, accelerate = self.instinct.evaluate(self, self.detected_objects) if self.instinct else (0.0, 0.0)

        self.turn(Angle(AngularType.RADIANS, turn))
        self.accelerate(accelerate)

    def _get_sensor_render(self) -> np.ndarray:
        """returns nx2 numpy array of vertices for the sensor shape."""
        half_fov = self.sense.field_of_view.value / 2.0
        n_points = 12

        # Define angular span based on shape
        if self.sense.shape == SensorShape.OMNI:
            angles = np.linspace(0, 2 * np.pi, n_points)
        else:
            angles = np.linspace(-half_fov, half_fov, n_points)

        base_r = self.sense.range

        # Shape-specific radius profiles
        if self.sense.shape == SensorShape.CONE:
            radii = np.full_like(angles, base_r)
        elif self.sense.shape == SensorShape.CARDIOID:
            # Polar cardioid: r = R * (1 + cos(θ)) / 2
            radii = base_r * (1 + np.cos(angles)) / 2.0
        elif self.sense.shape == SensorShape.BILOBED:
            # Polar bi-lobed: r = R * (1 + cos(2θ)) / 2
            radii = base_r * (1 + np.cos(2 * angles)) / 2.0
        elif self.sense.shape == SensorShape.OMNI:
            radii = np.full_like(angles, base_r)
        else:
            radii = np.full_like(angles, base_r)

        # Polar -> Cartesian using arctan2(x, y) angle convention
        # In arctan2(x, y): angle=0 points in +y, angle=π/2 points in +x
        # Point at angle θ: (x, y) = (r*sin(θ), r*cos(θ))
        rel_x = radii * np.sin(angles)
        rel_y = radii * np.cos(angles)

        # assure that our angles are in radians
        self.heading.cast_as_radians()
        # Negate heading to account for renderer's y-coordinate flip
        heading_rad = -self.heading.value

        # Rotate by heading using standard 2D rotation matrix
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        world_rel_x = rel_x * cos_h - rel_y * sin_h
        world_rel_y = rel_x * sin_h + rel_y * cos_h

        base_vertices = np.column_stack([self.position[0] + world_rel_x,
                                         self.position[1] + world_rel_y])

        if self.sense.shape != SensorShape.OMNI:
            edge_angles = np.array([-half_fov, half_fov])
            edge_radii = np.full_like(edge_angles, base_r)

            # Same arctan2(x, y) convention for edges
            edge_rel_x = edge_radii * np.sin(edge_angles)
            edge_rel_y = edge_radii * np.cos(edge_angles)

            # Same rotation matrix for edges
            edge_world_rel_x = edge_rel_x * cos_h - edge_rel_y * sin_h
            edge_world_rel_y = edge_rel_x * sin_h + edge_rel_y * cos_h

            edge_vertices = np.column_stack([self.position[0] + edge_world_rel_x,
                                             self.position[1] + edge_world_rel_y])

            # Insert at start/end to maintain correct angular ordering for polygon rendering
            final_vertices = np.vstack([edge_vertices[0:1], base_vertices, edge_vertices[1:]])
        else:
            final_vertices = base_vertices

        return final_vertices


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    positions = []
    pointers = []
    bounding_boxes = []

    sense_obj = Sense(type=SensorType.SIGHT,
                      shape=SensorShape.CONE,
                      range=15,
                      field_of_view=Angle(type=AngularType.RADIANS, value=1.0),
                      noise=0.12,
                      xray=False,
                      falloff_exponent=1.0
                      )


    object_a = BaseObject(mass=1, position=(20, 0), size=(2, 2), facing_point = (5, 5))
    object_b = BaseObject(mass=1, position=(0, 20), size=(2, 2), facing_point=(-20, -25))

    in_world = [object_a, object_b]
    ent = Vehicle(
        mass=2,
        position=(0, 0),
        size=(5.0, 5.0),
        facing_point=(4.0, 0.0),
        sense=sense_obj
    )

    print(ent)
    print(f"Initial heading: {ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    plt.figure(figsize=(20, 20))
    positions.append(ent.position.copy())
    pointers.append(ent.facing_point.copy())
    bounding_boxes.append(ent.bounding_box.copy())
    seen_states = [ent.perceive(in_world)]

    # Turn & move animation loop
    for i in range(18):
        if i < 2:
            # Clockwise turns
            turn_angle = 0.2 if i < 5 else 0.25
        else:
            # Counter-clockwise turns
            turn_angle = -0.2 if i < 10 else -0.25

        _turn = Angle(AngularType.RADIANS, turn_angle)

        ent.turn(_turn)
        ent.move()



        positions.append(ent.position.copy())
        pointers.append(ent.facing_point.copy())
        bounding_boxes.append(ent.bounding_box.copy())
        seen_states.append(ent.perceive(in_world))

        print(f"Step {i}: pos={ent.position}, facing_point={ent.facing_point}, "
              f"heading={ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    # Visualization loop
    for i in range(len(positions)):
        ax = plt.subplot(5, 4, i + 1)

        pos = positions[i]
        h = pointers[i]
        bb = bounding_boxes[i]
        seen = seen_states[i]

        # Draw position dot
        plt.plot(pos[0], pos[1], marker="o", color="blue", markersize=20)

        # Draw facing_point dot
        plt.plot(h[0], h[1], marker=".", color="orange", markersize=10)

        # Draw connection line
        plt.plot([pos[0], h[0]], [pos[1], h[1]], alpha=0.5, linewidth=2)

        # seen -- controlled visual via alpha
        print(seen)
        for e_idx, _ob in enumerate(in_world):
            c = ["red", "purple"]
            ep = _ob.position
            if seen[0][e_idx]:
                plt.scatter(ep[0], ep[1], color=c[e_idx], s=35, alpha=1.0)

        # Draw bounding box corners
        if bb is not None:
            # BB should be (4, 2) array of corners
            plt.plot(bb[:, 0], bb[:, 1], color="red", linewidth=2)
            for j, (x, y) in enumerate(bb):
                plt.plot(x, y, marker="s", color="green", markersize=30, alpha=0.1)


        ax.set_title(f"Step {i + 1}: heading={ent.heading.value:.4f} rad")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

    plt.tight_layout()
    plt.show()

    plt.plot(positions, color="blue", alpha=0.66, markersize=20)
    plt.plot(pointers, color="orange", alpha=0.5, markersize=10)
    plt.show()
    print("\n=== Final State ===")
    print(f"Position: {ent.position}")
    print(f"Facing Point: {ent.facing_point}")
    print(f"Heading: {ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")
    print(f"Bounding Box Shape: {ent.bounding_box.shape}")