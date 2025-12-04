from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainglobe-ccf-translator")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .PointSet import PointSet as PointSet
from .VolumeSeries import VolumeSeries as VolumeSeries
from .Volume import Volume as Volume

__all__ = ["PointSet", "VolumeSeries", "Volume", "__version__"]
