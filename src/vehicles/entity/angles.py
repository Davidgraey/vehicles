"""
Angle calculations
holds the Angle object class
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum

DEGREE_TO_RAD = np.pi / 180.0
RAD_TO_DEGREE = 180.0 / np.pi
TWOPI = 2*np.pi

# --------
class AngularType(Enum):
    RADIANS = "radians"
    DEGREES = "degrees"

# --------
@dataclass(slots=True)
class Angle:
    type: AngularType
    value: float = 0.0

    def to_degrees(self) -> np.ndarray:
        if self.type == AngularType.RADIANS:
            return self.value * RAD_TO_DEGREE
        return self.value

    def to_radians(self) -> np.ndarray:
        if self.type == AngularType.DEGREES:
            return self.value * DEGREE_TO_RAD
        return self.value

    def cast_as_radians(self) -> None:
        if self.type == AngularType.DEGREES:
            self.value *= DEGREE_TO_RAD
            self.type = AngularType.RADIANS
            print(self)

    def cast_as_degrees(self) -> None:
        if self.type == AngularType.RADIANS:
            self.value *= RAD_TO_DEGREE
            self.type = AngularType.DEGREES
            print(self)

    def __repr__(self):
        return f"Angle({self.type.value}, {self.value:.4f})"

    def __sub__(self, other: Angle):
        if self.type == other.type:
            return normalize_angle(
                Angle(type=self.type, value=self.value - other.value)
            )

        else:
            if self.type == AngularType.DEGREES:
                return normalize_angle(
                    Angle(type=AngularType.DEGREES,
                          value=self.value - other.to_degrees()
                          )
                )
            elif self.type == AngularType.RADIANS:
                return normalize_angle(
                    Angle(type=AngularType.RADIANS,
                          value=self.value - other.to_radians()
                          )
                )

    def __add__(self, other: Angle):
        if self.type == other.type:
            return normalize_angle(
                Angle(type=self.type, value=self.value + other.value)
            )

        else:
            if self.type == AngularType.DEGREES:
                return normalize_angle(
                    Angle(type=AngularType.DEGREES,
                          value=self.value + other.to_degrees()
                          )
                )
            elif self.type == AngularType.RADIANS:
                return normalize_angle(
                    Angle(type=AngularType.RADIANS,
                          value=self.value + other.to_radians()
                          )
                )

# ------------------------
def cartesian_motion_to_angular(cartesian_delta: tuple) -> tuple[Angle, np.ndarray | float]:
    """
    Converts an x,y point into angular motion (degrees) and speed
    Parameters
    ----------
    cartesian_delta : point difference, motion translated into (delta_x, delta_y))

    Returns
    -------
    Tuple of (heading, speed), wherein heading is the degrees
    """
    # unpack x and y values
    x,y = cartesian_delta
    speed = np.sqrt(x ** 2 + y ** 2)
    heading = np.arctan2(x, y)
    return (Angle(type=AngularType.RADIANS, value=heading), speed)


def angular_motion_to_cartesian(heading: Angle, speed: float) -> np.ndarray:
    """
    Converts the angular motion (heading (angle radians) and speed (float value)) into horizontal and vertical
    NOTE::: NOT an inverse to cartesian-to-angular funcitons!
    Parameters
    ----------
    heading : float, angular heading or facing in radians
    speed : float, current velocity/speed

    Returns
    -------
    tuple (x_delta, y_delta) - the total change that an object traveling at SPEED along HEADING angle translated into
    cartesian (x,y) coordiantes
    """
    x_delta = speed * np.sin(heading.value)
    y_delta = speed * np.cos(heading.value)

    return np.array([x_delta, y_delta])

def normalize_angle(angle:Angle) -> float:
    """Normalize angle to (-pi, pi], or -180 to 180"""
    # Wrap everything to [0, 2pi]
    if angle.type == AngularType.RADIANS:
        rotated = (angle.value + np.pi) % TWOPI
        rotated -= np.pi
    elif angle.type == AngularType.DEGREES:
        rotated = (angle.value + 180) % 360
        rotated -= 180

    return Angle(angle.type, rotated)


def calculate_relative_angle(sensor_heading: Angle, target_angle: Angle) -> float:
    """
    Calculate the signed angle difference between sensor heading and target.
    Returns angle in range [-pi, pi].
    """
    return normalize_angle(target_angle - sensor_heading)


if __name__ == "__main__":
    a = Angle(type=AngularType.DEGREES, value = 194.3)
    b = Angle(type=AngularType.RADIANS, value=2)
    print(a)
    print(b)
    print(a.type == b.type)
    c = a + b
    print(c)

    d = b - a
    print(d)

    print(normalize_angle(Angle(type=AngularType.DEGREES, value = 90*3)))

    print(normalize_angle(Angle(type=AngularType.RADIANS, value=5.2)))