"""
Static terrain: shaped regions that senses can observe and that can block movement, but that are not entities
(no mass, heading, health or behavior).

Shapes are numpy geometry; a Terrain pairs a shape with a kind for future interaction effects
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


def nearest_on_segments(points: np.ndarray, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    points : (M, 2)
    segments : (E, 2, 2) start and end point of each segment

    Returns
    -------
    distances : (M,) distance to the nearest segment
    nearest : (M, 2) nearest point on any segment
    edges : (M,) index of the nearest segment
    """
    start, end = segments[:, 0], segments[:, 1]
    edge = end - start
    length_sq = np.maximum((edge ** 2).sum(axis=-1), 1e-12)
    along = np.clip(((points[:, None, :] - start) * edge).sum(axis=-1) / length_sq, 0.0, 1.0)
    closest = start + along[..., None] * edge
    distances = np.linalg.norm(points[:, None, :] - closest, axis=-1)

    rows = np.arange(len(points))
    edges = distances.argmin(axis=1)
    return distances[rows, edges], closest[rows, edges], edges


def edge_perpendiculars(segments: np.ndarray) -> np.ndarray:
    """Unit vectors perpendicular to each (E, 2, 2) segment."""
    edge = segments[:, 1] - segments[:, 0]
    length = np.maximum(np.linalg.norm(edge, axis=-1, keepdims=True), 1e-12)
    direction = edge / length
    return np.stack([-direction[:, 1], direction[:, 0]], axis=-1)


def unit_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize (M, 2) vectors, using `fallback` where the length is zero."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norm > 1e-9
    return np.where(valid, vectors / np.where(valid, norm, 1.0), fallback)


def sample_segments(segments: np.ndarray, spacing: float) -> np.ndarray:
    """Points along each (E, 2, 2) segment, at most `spacing` apart; end points excluded."""
    samples = []
    for start, end in segments:
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
        fractions = np.arange(count) / count
        samples.append(start + fractions[:, None] * (end - start))
    return np.vstack(samples)


class Geometry(ABC):
    """
    Geometry of one terrain region, in world coordinates.

    Attributes
    ----------
    points : (P, 2) points that define the shape
    thickness : stroke width of an open shape; 0 for a closed region
    closed : True when `points` is a closed boundary
    """
    points: np.ndarray
    thickness: float
    closed: bool

    @abstractmethod
    def contact(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        points : (M, 2)

        Returns
        -------
        signed_distances : (M,) distance to the surface, negative inside
        normals : (M, 2) unit vectors pointing from the surface toward free space
        """

    @abstractmethod
    def boundary_samples(self, spacing: float) -> np.ndarray:
        """(K, 2) points on the boundary, at most `spacing` apart."""


class Polygon(Geometry):
    """
    Closed region bounded by straight edges; may be convex or concave.

    Parameters
    ----------
    points : (P, 2), at least three, in order around the boundary
    """
    thickness = 0.0
    closed = True

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        assert len(self.points) >= 3
        self.segments = np.stack([self.points, np.roll(self.points, -1, axis=0)], axis=1)

    @classmethod
    def rectangle(cls, center: Tuple[float, float], size: Tuple[float, float]) -> "Polygon":
        cx, cy = center
        half_w, half_h = size[0] / 2.0, size[1] / 2.0
        return cls([(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
                    (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)])

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Even-odd containment for (M, 2) points."""
        x, y = points[:, 0:1], points[:, 1:2]
        x1, y1 = self.segments[:, 0, 0], self.segments[:, 0, 1]
        x2, y2 = self.segments[:, 1, 0], self.segments[:, 1, 1]

        crosses = (y1 > y) != (y2 > y)
        dy = np.where(y2 == y1, 1.0, y2 - y1)
        crossing_x = x1 + (y - y1) * (x2 - x1) / dy
        return ((crosses & (x < crossing_x)).sum(axis=1) % 2) == 1

    def contact(self, points):
        distances, nearest, edges = nearest_on_segments(points, self.segments)
        inside = self.contains(points)

        away = np.where(inside[:, None], nearest - points, points - nearest)
        normals = unit_vectors(away, edge_perpendiculars(self.segments)[edges])
        return np.where(inside, -distances, distances), normals

    def boundary_samples(self, spacing):
        return sample_segments(self.segments, spacing)


class Wall(Geometry):
    """
    Thick polyline with rounded ends and joints.

    Parameters
    ----------
    points : (P, 2), at least two, along the wall's centerline
    thickness : full width of the wall
    """
    closed = False

    def __init__(self, points, thickness: float = 6.0):
        self.points = np.asarray(points, dtype=np.float64)
        assert len(self.points) >= 2
        self.thickness = thickness
        self.half = thickness / 2.0
        self.segments = np.stack([self.points[:-1], self.points[1:]], axis=1)

    def contact(self, points):
        distances, nearest, edges = nearest_on_segments(points, self.segments)
        normals = unit_vectors(points - nearest, edge_perpendiculars(self.segments)[edges])
        return distances - self.half, normals

    def boundary_samples(self, spacing):
        perpendiculars = edge_perpendiculars(self.segments)[:, None, :] * self.half
        edge = self.segments[-1, 1] - self.segments[-1, 0]
        first_edge = self.segments[0, 1] - self.segments[0, 0]
        caps = np.array([
            self.points[0] - self.half * first_edge / max(np.linalg.norm(first_edge), 1e-12),
            self.points[-1] + self.half * edge / max(np.linalg.norm(edge), 1e-12),
        ])
        return np.vstack([
            sample_segments(self.segments + perpendiculars, spacing),
            sample_segments(self.segments - perpendiculars, spacing),
            caps,
        ])


class Terrain:
    """
    A shaped region of the world.

    Parameters
    ----------
    shape : Geometry
    color : draw color
    sample_spacing : maximum gap between boundary points that senses can detect
    """
    blocks_movement = False

    def __init__(self, shape: Geometry, color: str = "gray", sample_spacing: float = 4.0):
        self.shape = shape
        self.color = color
        self.samples = shape.boundary_samples(sample_spacing)


class Barrier(Terrain):
    """Solid terrain that entities cannot move through."""
    blocks_movement = True

    def __init__(self, shape: Geometry, color: str = "dimgray", sample_spacing: float = 4.0):
        super().__init__(shape, color=color, sample_spacing=sample_spacing)
