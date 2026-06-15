"""fraudvae - hybrid (VAE novelty + boosted trees) fraud screening on a merged public transaction corpus."""
import os as _os
import sys as _sys

__version__ = "0.5.0"

# macOS: the PyTorch wheel ships its own copy of the LLVM OpenMP runtime and the
# LightGBM wheel links the homebrew one (brew install libomp). Two copies in one
# process make libomp abort the whole process the moment the second one starts
# ("OMP: Error #15"), which is exactly what happens when the server loads a run
# with a booster. KMP_DUPLICATE_LIB_OK is the runtime's own escape hatch: both
# copies then coexist, each library only ever drives its own thread pool. It has
# to be in the environment before either runtime initialises, so it lives here,
# at package import, rather than next to the lightgbm import.
if _sys.platform == "darwin":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
