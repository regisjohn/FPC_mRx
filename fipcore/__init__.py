from importlib.metadata import version, PackageNotFoundError
# metadata
try:
    __version__ = version("fipcore")
except PackageNotFoundError:
    __version__ = "0.0.0"
__author__ = "Regis John"

# Analysis
from .analysis.evec_proc import *
from .analysis.fpc_proc import *
from .analysis.fpc_mrx_main import *
from .analysis.jvec_proc import *
from .analysis.jdotE_proc import *
from .analysis.vdf_proc import *
from .analysis.vel_proc import *
from .analysis.params_proc import *

# Plotting
from .plotting.imagecont_fpc import *
from .plotting.imagecont_vdf import *
from .plotting.imagecont_vdf_fpc import *
from .plotting.plt_vmap_check import *
from .plotting.rtplot import *

# Utils
from .utils.coord_utils import *
from .utils.helper_utils import *
from .utils.io_utils import *


__all__ = [
    "evec_proc", "fpc_proc", "fpc_mrx_main", "jvec_proc", "jdotE_proc", 
    "vdf_proc", "vel_proc", "params_proc", "imagecont_fpc", "imagecont_vdf", 
    "imagecont_vdf_fpc", "plt_vmap_check", "rtplot", "coord_utils", 
    "helper_utils", "io_utils"
]