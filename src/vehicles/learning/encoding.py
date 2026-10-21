"""
Encode one Frame into per-vehicle rows for a single species.

Each vehicle gets a self row of S normalized floats and a grid row of R*A tokens: its sense's area
unwrapped into rings (distance / range) and sectors (position along the sense's angular extent).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from vehicles.entity.base_object import BaseObject
from vehicles.entity.senses import Sense
from vehicles.learning.tokens import Token, token_for
from vehicles.world.terrain import Terrain
from vehicles.world.world_view import Frame

EDGE_FEATURES = ("edge_left", "edge_right", "edge_bottom", "edge_top")
FEATURES = ("velocity", "hunger", "health", "last_turn", "last_accelerate", "heading_sin", "heading_cos") + EDGE_FEATURES


@dataclass(frozen=True)
class EncodingSpec:
    """
    Parameters
    ----------
    history : T, ticks per sample
    rings : R, distance bins
    sectors : A, angular bins
    self_features : names from FEATURES, in index order
    """
    history: int = 16
    rings: int = 12
    sectors: int = 24
    self_features: Tuple[str, ...] = ("velocity", "hunger", "health", "last_turn", "last_accelerate")

    @property
    def self_dim(self) -> int:
        return len(self.self_features)

    @property
    def grid_start(self) -> int:
        return self.self_dim

    @property
    def cells(self) -> int:
        return self.rings * self.sectors

    @property
    def dim(self) -> int:
        return self.self_dim + self.cells

    @property
    def vocab_size(self) -> int:
        return len(Token)

    @property
    def feature_names(self) -> List[str]:
        cells = [f"ring{r}_sector{a}" for r in range(self.rings) for a in range(self.sectors)]
        return list(self.self_features) + cells


def grid_cells(sense: Sense, distances: np.ndarray, relative_angles: np.ndarray,
               rings: int, sectors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin targets into the sense's unwrapped area.

    Parameters
    ----------
    sense : the observing sense
    distances, relative_angles : (M,) as measured from the observer
    rings, sectors : grid size

    Returns
    -------
    ring, sector : (M,) int; sector is -1 for targets outside the sense's angular extent
    """
    ring = np.clip(np.floor(distances / sense.range * rings).astype(int), 0, rings - 1)

    intervals = sense.angular_extent()
    widths = np.array([end - start for start, end in intervals])
    offsets = np.concatenate([[0.0], np.cumsum(widths)[:-1]])

    position = np.full(len(relative_angles), np.nan)
    for (start, _), width, offset in zip(intervals, widths, offsets):
        along = (relative_angles - start) % (2 * np.pi)
        inside = np.isnan(position) & (along <= width)
        position[inside] = offset + along[inside]

    outside = np.isnan(position)
    sector = np.clip(np.floor(np.nan_to_num(position) / widths.sum() * sectors).astype(int), 0, sectors - 1)
    sector[outside] = -1
    return ring, sector


def resolve_cells(cells: np.ndarray, layers: np.ndarray, distances: np.ndarray,
                  tokens: np.ndarray, size: int) -> np.ndarray:
    """
    One token per cell: lowest layer first (entities 0, terrain 1), then nearest.

    Returns
    -------
    (size,) int8, 0 where nothing was detected
    """
    grid = np.zeros(size, dtype=np.int8)
    if len(cells) == 0:
        return grid

    order = np.lexsort((distances, layers, cells))
    _, first = np.unique(cells[order], return_index=True)
    picked = order[first]
    grid[cells[picked]] = tokens[picked]
    return grid


class ObservationEncoder:
    """
    Encodes Frames for every entity of exactly one species.

    Parameters
    ----------
    species : Vehicle class; entities whose type is exactly this are encoded
    entities : every entity in the world, in Frame order
    terrain : every terrain region in the world, in Frame order
    spec : EncodingSpec
    """
    def __init__(self, species: type, entities: List[BaseObject], terrain: List[Terrain], spec: EncodingSpec = EncodingSpec()):
        unknown = set(spec.self_features) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown self features: {sorted(unknown)}")

        self.species = species
        self.spec = spec
        self.rows = np.array([i for i, e in enumerate(entities) if type(e) is species], dtype=np.int64)
        if len(self.rows) == 0:
            raise ValueError(f"No {species.__name__} in the world")

        members = [entities[i] for i in self.rows]
        self.senses = [m.sense for m in members]
        if len({s.shape for s in self.senses}) > 1:
            raise ValueError(f"{species.__name__} senses do not share one shape")

        self.ids = np.array([e.id for e in entities], dtype=np.int64)
        self.entity_tokens = np.array([token_for(type(e)) for e in entities], dtype=np.int8)
        self.terrain_tokens = np.concatenate(
            [np.full(len(t.samples), token_for(type(t)), dtype=np.int8) for t in terrain]
        ) if terrain else np.empty(0, dtype=np.int8)

        self.max_speed = np.array([m.max_speed for m in members], dtype=np.float64)
        self.starvation_threshold = np.array([m.starvation_threshold for m in members], dtype=np.float64)
        self.max_health = np.array([m.max_health for m in members], dtype=np.float64)
        self.sense_range = np.array([s.range for s in self.senses], dtype=np.float64)
        self.world_bounds = None
        if set(spec.self_features) & set(EDGE_FEATURES):
            self.world_bounds = np.array([m.world_bounds for m in members], dtype=np.float64)

        self.latest_tick: Optional[int] = None
        self.latest_self_block: Optional[np.ndarray] = None
        self.latest_grid: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    def check_layout(self, frame: Frame) -> None:
        """Raise if the Frame's entities or terrain differ from those the encoder was built with."""
        if not np.array_equal(frame.ids, self.ids) or len(frame.terrain_samples) != len(self.terrain_tokens):
            raise ValueError("World layout changed since the encoder was built")

    def self_block(self, frame: Frame) -> np.ndarray:
        """(B, S) float32 self features."""
        rows = self.rows
        columns = {
            "velocity": frame.velocities[rows] / self.max_speed,
            "hunger": frame.hunger[rows] / self.starvation_threshold,
            "health": frame.health[rows] / self.max_health,
            "last_turn": frame.previous_actions[rows, 0],
            "last_accelerate": frame.previous_actions[rows, 1],
            "heading_sin": np.sin(frame.headings[rows]),
            "heading_cos": np.cos(frame.headings[rows]),
        }
        if self.world_bounds is not None:
            x, y = frame.positions[rows, 0], frame.positions[rows, 1]
            columns["edge_left"] = x / self.sense_range
            columns["edge_right"] = (self.world_bounds[:, 0] - x) / self.sense_range
            columns["edge_bottom"] = y / self.sense_range
            columns["edge_top"] = (self.world_bounds[:, 1] - y) / self.sense_range

        return np.stack([columns[name] for name in self.spec.self_features], axis=1).astype(np.float32)

    def grid(self, frame: Frame) -> np.ndarray:
        """(B, R*A) int8 tokens."""
        rings, sectors = self.spec.rings, self.spec.sectors
        grid = np.zeros((self.batch_size, self.spec.cells), dtype=np.int8)

        for b, (row, sense) in enumerate(zip(self.rows, self.senses)):
            seen = np.flatnonzero(frame.detected[row])
            seen_terrain = np.flatnonzero(frame.terrain_detected[row])

            distances = np.concatenate([frame.distances[row, seen], frame.terrain_distances[row, seen_terrain]])
            angles = np.concatenate([frame.relative_angles[row, seen], frame.terrain_relative_angles[row, seen_terrain]])
            tokens = np.concatenate([self.entity_tokens[seen], self.terrain_tokens[seen_terrain]])
            layers = np.concatenate([np.zeros(len(seen), dtype=np.int8), np.ones(len(seen_terrain), dtype=np.int8)])

            ring, sector = grid_cells(sense, distances, angles, rings, sectors)
            inside = sector >= 0
            cells = ring * sectors + sector
            grid[b] = resolve_cells(cells[inside], layers[inside], distances[inside], tokens[inside], self.spec.cells)

        return grid

    def encode(self, frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        self_block : (B, S) float32
        grid : (B, R*A) int8
        """
        self.check_layout(frame)
        self.latest_tick = frame.tick
        self.latest_self_block = self.self_block(frame)
        self.latest_grid = self.grid(frame)
        return self.latest_self_block, self.latest_grid

    def __str__(self) -> str:
        """The latest encoding, per vehicle: self features, then the grid as rings (rows) by sectors (columns)."""
        name, rings, sectors = self.species.__name__, self.spec.rings, self.spec.sectors
        header = f"ObservationEncoder({name}, B={self.batch_size}, R={rings}, A={sectors})"
        if self.latest_grid is None:
            return f"{header}: nothing encoded yet"

        legend = ", ".join(f"{t.value}={t.name}" for t in Token if t)
        lines = [f"{header}, tick {self.latest_tick}", f"tokens: {legend}"]
        for b, row in enumerate(self.rows):
            lines.append(f"{name} {b} (id {self.ids[row]})")
            lines += [f"  {feature:>16} {value: .3f}" for feature, value in zip(self.spec.self_features, self.latest_self_block[b])]
            lines += ["  " + " ".join(f"{v:2d}" if v else " ." for v in ring) for ring in self.latest_grid[b].reshape(rings, sectors)]
        return "\n".join(lines)
