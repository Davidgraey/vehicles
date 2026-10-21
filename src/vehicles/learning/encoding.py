"""
Encode one Frame or Tick - either with:

ObservationEncoder: per-vehicle rows for one species, a self row of S normalized floats and a grid row of R*A
tokens (the vehicle's sense area unwrapped into rings and sectors).

WorldEncoder: one omniscient view of every entity and all terrain, as an entity table and a raster of cells
holding an entity token, a terrain token and continuous planes.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from vehicles.entity.base_object import BaseObject
from vehicles.entity.senses import Sense
from vehicles.entity.vehicle import Vehicle
from vehicles.learning.tokens import Token, token_lookup
from vehicles.world.terrain import Terrain
from vehicles.world.world_view import Frame

EDGE_FEATURES = ("edge_left", "edge_right", "edge_bottom", "edge_top")
BODY_FEATURES = ("velocity", "health", "heading_sin", "heading_cos")
VEHICLE_FEATURES = ("hunger", "last_turn", "last_accelerate")
FEATURES = BODY_FEATURES + VEHICLE_FEATURES + EDGE_FEATURES
GEOMETRY_PLANES = ("dx", "dy", "heading_sin", "heading_cos")
GLYPHS = {Token.TERRAIN: "+", Token.BARRIER: "#"}


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


@dataclass(frozen=True)
class WorldSpec:
    """
    Parameters
    ----------
    width, height : world size in units
    cell_size : side of one raster cell in units
    features : names from FEATURES, excluding the edge features, in plane order
    """
    width: float = 800.0
    height: float = 600.0
    cell_size: float = 8.0
    features: Tuple[str, ...] = ("velocity", "health", "hunger", "last_turn", "last_accelerate")

    @property
    def columns(self) -> int:
        return int(np.ceil(self.width / self.cell_size))

    @property
    def rows(self) -> int:
        return int(np.ceil(self.height / self.cell_size))

    @property
    def cells(self) -> int:
        return self.rows * self.columns

    @property
    def planes(self) -> int:
        return len(GEOMETRY_PLANES) + len(self.features)

    @property
    def plane_names(self) -> List[str]:
        return list(GEOMETRY_PLANES) + list(self.features)


@dataclass(frozen=True)
class WorldTable:
    """
    One tick of every entity, indexed by entity (N).

    Parameters
    ----------
    tick : tick of the Frame
    tokens : (N,) int8
    cell : (N,) int32 flat raster cell, after clipping to the world
    offset : (N, 2) float32 position within the cell, in [-0.5, 0.5]
    heading : (N, 2) float32 sin, cos
    features : (N, F) float32 normalized; 0 where absent
    present : (N, F) bool; False where the entity has no such feature
    kept : (N,) bool; False where another entity won the cell
    clipped : entities outside the world, moved to the edge cell
    """
    tick: int
    tokens: np.ndarray
    cell: np.ndarray
    offset: np.ndarray
    heading: np.ndarray
    features: np.ndarray
    present: np.ndarray
    kept: np.ndarray
    clipped: int

    @property
    def dropped(self) -> int:
        return int((~self.kept).sum())


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


def pick_per_cell(cells: np.ndarray, distances: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """
    Index of the winner in each occupied cell: lowest layer first, then nearest.

    Parameters
    ----------
    cells, distances, layers : (M,)

    Returns
    -------
    (U,) indices into the inputs, one per distinct cell
    """
    order = np.lexsort((distances, layers, cells))
    _, first = np.unique(cells[order], return_index=True)
    return order[first]


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

    picked = pick_per_cell(cells, distances, layers)
    grid[cells[picked]] = tokens[picked]
    return grid


def rasterize_terrain(terrain: List[Terrain], spec: WorldSpec) -> np.ndarray:
    """
    Mark every cell whose center lies in a terrain region.

    An open shape marks the cells its stroke crosses even when thinner than a cell. Barrier wins over terrain.

    Returns
    -------
    (rows, columns) int8 of Token.EMPTY, Token.TERRAIN or Token.BARRIER
    """
    size = spec.cell_size
    columns, rows = np.meshgrid(np.arange(spec.columns), np.arange(spec.rows))
    centers = (np.stack([columns, rows], axis=-1).reshape(-1, 2) + 0.5) * size

    raster = np.zeros(spec.cells, dtype=np.int8)
    for region in sorted(terrain, key=lambda t: t.blocks_movement):
        signed, _ = region.shape.contact(centers)
        reach = 0.0 if region.shape.closed else max(0.0, (size - region.shape.thickness) / 2.0)
        raster[signed <= reach] = token_lookup(type(region))
    return raster.reshape(spec.rows, spec.columns)


class Encoder:
    """
    What both encoders share: the world layout, entity tokens and normalized per-entity features.

    Parameters
    ----------
    entities : every entity in the world, in Frame order
    terrain : every terrain region in the world, in Frame order
    """
    def __init__(self, entities: List[BaseObject], terrain: List[Terrain]):
        self.ids = np.array([e.id for e in entities], dtype=np.int64)
        self.entity_tokens = np.array([token_lookup(type(e)) for e in entities], dtype=np.int8)
        self.sample_count = sum(len(t.samples) for t in terrain)

        self.is_vehicle = np.array([isinstance(e, Vehicle) for e in entities], dtype=bool)
        self.max_speed = np.array([e.max_speed for e in entities], dtype=np.float64)
        self.max_health = np.array([e.max_health for e in entities], dtype=np.float64)
        self.starvation_threshold = np.array([getattr(e, "starvation_threshold", 1.0) for e in entities], dtype=np.float64)

    @staticmethod
    def check_features(names: Tuple[str, ...], allowed: Tuple[str, ...] = FEATURES) -> None:
        """Raise if any name is not in `allowed`."""
        unknown = set(names) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")

    def check_layout(self, frame: Frame) -> None:
        """Raise if the Frame's entities or terrain differ from those the encoder was built with."""
        if not np.array_equal(frame.ids, self.ids) or len(frame.terrain_samples) != self.sample_count:
            raise ValueError("World layout changed since the encoder was built")

    def present(self, names: Tuple[str, ...]) -> np.ndarray:
        """(N, len(names)) bool, False where an entity lacks the feature."""
        return np.stack([self.is_vehicle if name in VEHICLE_FEATURES else np.ones_like(self.is_vehicle) for name in names], axis=1)

    def columns(self, frame: Frame) -> Dict[str, np.ndarray]:
        """Every body and vehicle feature of every entity as an (N,) float64 column; 0 where absent."""
        return {
            "velocity": frame.velocities / np.maximum(self.max_speed, 1e-9),
            "health": frame.health / np.maximum(self.max_health, 1e-9),
            "heading_sin": np.sin(frame.headings),
            "heading_cos": np.cos(frame.headings),
            "hunger": frame.hunger / np.maximum(self.starvation_threshold, 1e-9),
            "last_turn": frame.previous_actions[:, 0],
            "last_accelerate": frame.previous_actions[:, 1],
        }


class ObservationEncoder(Encoder):
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
        self.check_features(spec.self_features)
        super().__init__(entities, terrain)

        self.species = species
        self.spec = spec
        self.rows = np.array([i for i, e in enumerate(entities) if type(e) is species], dtype=np.int64)
        if len(self.rows) == 0:
            raise ValueError(f"No {species.__name__} in the world")

        members = [entities[i] for i in self.rows]
        self.senses = [m.sense for m in members]
        if len({s.shape for s in self.senses}) > 1:
            raise ValueError(f"{species.__name__} senses do not share one shape")

        self.terrain_tokens = np.concatenate(
            [np.full(len(t.samples), token_lookup(type(t)), dtype=np.int8) for t in terrain]
        ) if terrain else np.empty(0, dtype=np.int8)

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

    def self_block(self, frame: Frame) -> np.ndarray:
        """(B, S) float32 self features."""
        columns = {name: column[self.rows] for name, column in self.columns(frame).items()}
        if self.world_bounds is not None:
            x, y = frame.positions[self.rows, 0], frame.positions[self.rows, 1]
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


class WorldEncoder(Encoder):
    """
    Encodes Frames of the whole world.

    Parameters
    ----------
    entities : every entity in the world, in Frame order
    terrain : every terrain region in the world, in Frame order
    spec : WorldSpec
    """
    def __init__(self, entities: List[BaseObject], terrain: List[Terrain], spec: WorldSpec = WorldSpec()):
        self.check_features(spec.features, BODY_FEATURES + VEHICLE_FEATURES)
        super().__init__(entities, terrain)

        self.spec = spec
        self.present_features = self.present(spec.features)
        self.terrain_raster = rasterize_terrain(terrain, spec)
        self.latest: Optional[WorldTable] = None

    def encode(self, frame: Frame) -> WorldTable:
        """Encode one Frame; the result is also kept as `latest`."""
        self.check_layout(frame)
        spec = self.spec

        scaled = frame.positions / spec.cell_size
        column = np.floor(scaled[:, 0]).astype(int)
        row = np.floor(scaled[:, 1]).astype(int)
        inside = (column >= 0) & (column < spec.columns) & (row >= 0) & (row < spec.rows)
        column = np.clip(column, 0, spec.columns - 1)
        row = np.clip(row, 0, spec.rows - 1)
        cell = row * spec.columns + column
        offset = np.clip(scaled - np.stack([column, row], axis=1) - 0.5, -0.5, 0.5)

        kept = np.zeros(len(cell), dtype=bool)
        kept[pick_per_cell(cell, np.linalg.norm(offset, axis=1), np.zeros(len(cell), dtype=np.int8))] = True

        columns = self.columns(frame)
        features = np.stack([columns[name] for name in spec.features], axis=1) * self.present_features
        heading = np.stack([columns["heading_sin"], columns["heading_cos"]], axis=1)

        self.latest = WorldTable(
            tick=frame.tick,
            tokens=self.entity_tokens,
            cell=cell.astype(np.int32),
            offset=offset.astype(np.float32),
            heading=heading.astype(np.float32),
            features=features.astype(np.float32),
            present=self.present_features,
            kept=kept,
            clipped=int((~inside).sum()),
        )
        return self.latest

    def dense(self, table: WorldTable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        entity_tokens : (rows, columns) int8, 0 where no entity
        planes : (P, rows, columns) float32, 0 where no entity
        """
        spec = self.spec
        kept = table.kept
        entity_tokens = np.zeros(spec.cells, dtype=np.int8)
        planes = np.zeros((spec.planes, spec.cells), dtype=np.float32)

        entity_tokens[table.cell[kept]] = table.tokens[kept]
        values = np.concatenate([table.offset, table.heading, table.features], axis=1)
        planes[:, table.cell[kept]] = values[kept].T
        return entity_tokens.reshape(spec.rows, spec.columns), planes.reshape(spec.planes, spec.rows, spec.columns)

    def render(self, stride: int = 1) -> str:
        """The latest raster as characters: entity initial, `+` terrain, `#` barrier, `.` empty."""
        entity_tokens, _ = self.dense(self.latest)
        rows, columns = entity_tokens.shape
        pad = ((0, -rows % stride), (0, -columns % stride))

        def pool(grid: np.ndarray) -> np.ndarray:
            grid = np.pad(grid, pad)
            blocks = grid.reshape(grid.shape[0] // stride, stride, grid.shape[1] // stride, stride)
            return blocks.transpose(0, 2, 1, 3).reshape(grid.shape[0] // stride, grid.shape[1] // stride, -1).max(axis=-1)

        entities, terrain = pool(entity_tokens), pool(self.terrain_raster)
        shown = np.where(entities > 0, entities, terrain)
        return "\n".join(
            "".join("." if t == 0 else GLYPHS.get(Token(t), Token(t).name[0]) for t in line) for line in shown
        )

    def __str__(self) -> str:
        spec = self.spec
        header = f"WorldEncoder(N={len(self.ids)}, {spec.rows}x{spec.columns} cells of {spec.cell_size:g}, P={spec.planes})"
        if self.latest is None:
            return f"{header}: nothing encoded yet"

        stride = int(np.ceil(spec.columns / 100))
        legend = ", ".join(f"{t.name[0]}={t.name}" for t in Token if 0 < t < Token.TERRAIN)
        return "\n".join([
            f"{header}, tick {self.latest.tick}, stride {stride}",
            f"{legend}, +=TERRAIN, #=BARRIER",
            f"dropped {self.latest.dropped}, clipped {self.latest.clipped}",
            self.render(stride),
        ])


if __name__ == "__main__":
    from vehicles.entity.species import Fish, Predator
    from vehicles.world.terrain import Barrier, Polygon, Wall
    from vehicles.world.world import World

    world = World(width=320, height=240, seed=0)
    world.add_entity([Fish(position=(60 + 20 * i, 120), facing_point=(60 + 20 * i, 130)) for i in range(5)])
    world.add_entity(Predator(position=(250, 60), facing_point=(250, 70)))
    world.add_terrain([
        Barrier(Wall([(40, 190), (260, 190)], thickness=6)),
        Terrain(Polygon.rectangle(center=(200, 110), size=(60, 36))),
    ])

    observer = ObservationEncoder(Fish, world.entities, world.terrain)
    overview = WorldEncoder(world.entities, world.terrain, WorldSpec(width=world.width, height=world.height))

    for _ in range(5):
        world.step()
        observer.encode(world.frame)
        overview.encode(world.frame)

    print(overview)
    print(observer)
