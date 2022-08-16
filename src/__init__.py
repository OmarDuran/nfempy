
from . import geometry

__all__ = ["geometry", "integration", "mesh", "postprocess"]

__all__.extend(geometry.__all__)
__all__.extend(integration.__all__)
__all__.extend(mesh.__all__)
__all__.extend(postprocess.__all__)

submodules = [geometry, integration, mesh, postprocess]


try:
    nfempy_dir = os.path.dirname(__file__)
except:
    nfempy_dir = ""

__version__ = "0.0.0"
