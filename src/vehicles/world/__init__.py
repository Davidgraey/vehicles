"""
________ set up public objects ________
world obj iteelf is pygame-free and imported eagerly. Renderer/SimulationApp pull in pygame,
so they are loaded lazily to keep headless `import vehicles.world` cheap.
"""
from .world import World

__version__ = "0.1.0"
__all__ = ["__version__", "World", "Renderer"]

def __getattr__(name):
    if name == "Renderer":
        from .renderer import Renderer
        return Renderer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
