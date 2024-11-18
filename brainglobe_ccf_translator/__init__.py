from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainglobe-ccf-translator")
except PackageNotFoundError:
    # package is not installed
    pass

from . import *
from .PointSet import PointSet
from .VolumeSeries import VolumeSeries
from .Volume import Volume
