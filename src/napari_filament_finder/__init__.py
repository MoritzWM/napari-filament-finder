try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

from ._sample_data import make_sample_data
from ._widget import find_filaments

__all__ = (
    'make_sample_data',
    "find_filaments",
)
