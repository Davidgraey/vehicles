import numpy as np
import random
from vehicles.entity.angles import Angle, AngularType, angular_motion_to_cartesian
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass


def get_facing_angle(center_point: np.array, facing_point: np.array) -> Angle:
    """
    From the center point and directional indicator facing point, return the angle
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
                 facing_point: Tuple[int, int]
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
        self.velocity: np.ndarray = np.array([0.0, 0.0], dtype=np.float16)


        # Constants ----------------
        self.max_speed: float = 20.0  # 20 - defined by entity type
        self.speed: float = 3.0  # 2 - defined by entity type - also the acceleration

        # Constantly Updated Variables
        self.old_speed = (0, 0)
        self.heading = get_facing_angle(self.position, self.facing_point)
        self.heading.cast_as_radians()

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

    # ------------------------ MOVEMENT ------------------------
    def move(self) -> None:
        # move self.position by the x and y deltas
        _delta = angular_motion_to_cartesian(self.heading, self.speed)
        print("moving ", _delta, " units")
        self.position += _delta
        self.facing_point += _delta

        return None

    def turn(self, angular_movement: Angle) -> None:
        if isinstance(angular_movement, (float | int)):
            angular_movement = Angle(AngularType.RADIANS, angular_movement)

        self.heading = self.heading + angular_movement

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
    def _colliders(self) -> np.ndarray:
        _pos = self.bounding_box
        return np.array([
            np.min(_pos[:, 0]),
            np.min(_pos[:, 1]),
            np.max(_pos[:, 0]),
            np.max(_pos[:, 1])
        ])

    @property
    def direction(self) -> Angle:
        # angular direction in degrees
        return get_facing_angle(self.position, self.facing_point)

    def check_collision(self, instance_objects: list[BaseObject,]) -> np.ndarray:
        # minx, miny, maxx, maxy
        corners_1 = self._colliders
        target_positions = np.array([t.position for t in instance_objects])

        return np.array([
                corners_1[0] <= target_positions[:, 0]
                and corners_1[1] <= target_positions[:, 1]
                and corners_1[2] >= target_positions[:, 2]
                and corners_1[3] >= target_positions[:, 3]
        ])

    def __repr__(self):
        return f'Object at {self.position} facing {self.direction} \n has mass of {self.mass} and is size {self.size}'

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
