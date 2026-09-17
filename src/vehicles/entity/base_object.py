import numpy as np
import random
from vehicles.entity.angles import Angle, AngularType, angular_motion_to_cartesian
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass


def get_facing_angle(center_point: np.array, facing_point: np.array) -> Angle:
    """
    From the center point and directiget_facing_angleonal indicator facing point, return the angle
    Parameters
    ----------
    center_point : array, [x_coordinate, y_coordinate]
    facing_point : array, [x_coordinate, y_coordinate]

    Returns
    -------
    the angle of facing, in degrees, as a float
    """
    _delta = facing_point - center_point
    radian_angle = Angle(type=AngularType.RADIANS, value=np.arctan2(_delta[0], _delta[1]))

    return radian_angle


# ---------------------------------------------------------------------------------------------------
class BaseObject:
    """ Basal class of a gameobject that can move and do cool shit """
    def __init__(self,
                 mass: int,
                 position: Tuple[int, int],
                 size: Tuple[int, int],
                 facing_point: Tuple[int, int],
                 max_speed: Optional[float] = 5,
                 speed: Optional[float] = 0.2,
                 ):
        """
        Parameters
        ----------
        mass : mass of entity
        position : x, y coordinates of object's center(?) point
        size : width, height
        facing_point : x, y coordinates that represent the "facing" point; direction
        """
        self.id: int = random.randint(1, 9999)
        self.mass: float = mass
        assert len(size) == 2
        self.size: np.ndarray = np.array(size)

        # initialize position ----------------
        self.position: np.ndarray = np.array(position, dtype=np.float16)
        self.facing_point: np.ndarray = np.array(facing_point, dtype=np.float16)
        self.velocity: float = 0.0


        # Constants ----------------
        self.max_speed: float = max_speed
        self.speed: float = speed # accel rate

        # Constantly Updated Variables
        self.heading = get_facing_angle(self.position, self.facing_point)
        self.heading.cast_as_radians()
        self.colliding: list = []  # other BaseObjects currently touching this one

        # Appearance -- generic so the renderer never needs isinstance checks.
        # A subclass (see entity/species.py's Fish) overrides these after
        # calling super().__init__() to get its own look.
        self.color: str = "steelblue"

    # ------------------------ MOVEMENT ------------------------
    def move(self) -> None:
        # move self.position by the x and y deltas
        _delta = angular_motion_to_cartesian(self.heading, self.velocity)
        self.position += _delta
        self.facing_point += _delta

        return None

    def accelerate(self, direction: float) -> None:
        """
        Direction will be +1, 0 or -1 to tell us which way we're moving
        """
        if direction == 0:
            momentum = 0.95 #1 / self.mass
            self.velocity *= momentum

        else:
            momentum = direction * (self.speed / self.mass)
            self.velocity += momentum

        self.velocity = np.clip(
            self.velocity,
            a_min=-(self.max_speed / 2),
            a_max=self.max_speed
        )

    def turn(self, angular_movement: Angle) -> None:
        if isinstance(angular_movement, (float | int)):
            angular_movement = Angle(AngularType.RADIANS, angular_movement)

        self.heading = self.heading - angular_movement

        local_vector = self.facing_point - self.position

        theta = angular_movement.value
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        rotated_offset = (rotation_matrix @ local_vector.T).T # back to (4,2)
        self.facing_point = self.position + rotated_offset

        return None

    @property
    def bounding_box(self) -> np.ndarray:
        """
        Calculated Axis-Aligned Bounding Box (AABB) as a dict for compatibility
        or collision logic.
        """
        x, y = self.position
        w, h = self.size

        theta = self.heading.value  # guaranteed radians
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        # Calculate corners of the rect based on center/position and size
        rect = np.array(
            [
                [(x - w / 2)-x, (y - h / 2)-y], [(x - w / 2)-x, (y + h / 2)-y],
                [(x + w / 2)-x, (y - h / 2)-y], [(x + w / 2)-x, (y + h / 2)-y]
            ]
        )
        rotated_offsets = (rotation_matrix @ rect.T).T # back to (4,2)

        return self.position + rotated_offsets

    @property
    def radius(self) -> float:
        """
        Collision radius: a circle approximation of this object's footprint,
        half its longest side. Matches how the renderer already draws bodies
        (`radius = max(size) // 2`), so "touching" lines up with what's on
        screen.
        """
        return float(np.max(self.size)) / 2.0

    @property
    def direction(self) -> Angle:
        # angular direction in degrees
        return get_facing_angle(self.position, self.facing_point)

    def _get_body_render(self) -> np.ndarray | None:
        """
        Optional custom body silhouette for the renderer to draw instead of
        the default circle. None (the default here) means "just draw the
        plain circle" -- override in a subclass to return an Nx2 array of
        world-space polygon vertices instead. Build it the same way
        Vehicle._get_sensor_render() builds the sense cone: local offsets
        (forward = +y) rotated by heading using the renderer's y-flip
        convention (heading_rad = -self.heading.value), then translated by
        self.position.
        """
        return None

    def check_collision(self, instance_objects: list) -> np.ndarray:
        """
        Circle touch test against every other object
        """
        others = [obj for obj in instance_objects if obj != self]
        if not others:
            return np.array([], dtype=bool)

        other_positions = np.array([o.position for o in others], dtype=np.float64)
        other_radii = np.array([o.radius for o in others], dtype=np.float64)

        # float64 here on purpose: positions are stored float16 elsewhere, and
        # squaring a difference of a few hundred units overflows float16.
        distances = np.linalg.norm(other_positions - self.position.astype(np.float64), axis=1)

        return distances <= (self.radius + other_radii)

    def __repr__(self):
        return f'Object at {self.position} facing {self.direction} \n has mass of {self.mass} and is size {self.size}'


 # ------------------------  Visualize ------------------------
    # def render(self):
    #     surface = pg.Surface(self.size)
    #     surface = surface.fill(color=(0,0,0,0))
    #     surface = self._draw_sense(surface)
    #     surface = self._draw_self(surface)
    #
    #     # canvas.blit(self, self.position)
    #     # canvas.blit(sprites, (self.x_position, self.y_position))  # draw single sprite
    #     return surface, self.position
    #
    # def _draw_sense(self, surface):
    #     # composite and return the sensory objects
    #     # TODO: senses - build and draw
    #
    #     # example ----
    #     pg.draw.circle(surface=surface,
    #                    color=(255, 255, 0, 50),
    #                    center=self.position,
    #                    radius=max(self.size[0], self.size[1]) * 1.5,
    #                    width=0)
    #
    #     return surface
    #
    # def _draw_self(self, surface):
    #     # composite and return entity object
    #     # example ----
    #     pg.draw.circle(surface=surface,
    #                    color=(255, 255, 0, 125),
    #                    center=self.position,
    #                    radius=max(self.size[0], self.size[1]) // 2,
    #                    width=4, # width of > 0 - stroke
    #                    draw_top_left=True)
    #
    #     # surface.blit()
    #     return surface

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    positions = []
    pointers = []
    bounding_boxes = []


    ent = BaseObject(
        mass=2,
        position=(5.0, 5.0),
        size=(5.0, 5.0),
        facing_point=(10.0, 10.0),
    )

    print(ent)
    print(f"Initial heading: {ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    plt.figure(figsize=(20, 20))
    positions.append(ent.position.copy())
    pointers.append(ent.facing_point.copy())
    bounding_boxes.append(ent.bounding_box.copy())

    # Turn & move animation loop
    for i in range(18):
        if i < 2:
            # Clockwise turns
            turn_angle = 0.33 if i < 5 else 0.5
        else:
            # Counter-clockwise turns
            turn_angle = -0.33 if i < 10 else -0.5

        _turn = Angle(AngularType.RADIANS, turn_angle)

        ent.turn(_turn)
        ent.move()

        positions.append(ent.position.copy())
        pointers.append(ent.facing_point.copy())
        bounding_boxes.append(ent.bounding_box.copy())

        print(f"Step {i}: pos={ent.position}, facing_point={ent.facing_point}, "
              f"heading={ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    # Visualization loop
    for i in range(len(positions)):
        ax = plt.subplot(5, 4, i + 1)

        pos = positions[i]
        h = pointers[i]
        bb = bounding_boxes[i]

        # Draw position dot
        plt.plot(pos[0], pos[1], marker="o", color="blue", markersize=20)

        # Draw facing_point dot
        plt.plot(h[0], h[1], marker=".", color="orange", markersize=10)

        # Draw connection line
        plt.plot([pos[0], h[0]], [pos[1], h[1]], alpha=0.5, linewidth=2)

        # Draw bounding box corners
        if bb is not None:
            # BB should be (4, 2) array of corners
            plt.plot(bb[:, 0], bb[:, 1], color="red", linewidth=2)
            for j, (x, y) in enumerate(bb):
                plt.plot(x, y, marker="s", color="green", markersize=30, alpha=0.1)


        ax.set_title(f"Step {i + 1}: heading={ent.heading.value:.4f} rad")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

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
