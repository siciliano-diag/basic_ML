#from . import model_utils
from . import pipeline
from ._version import __version__

import os
print(os.getcwd())

__all__ = [#"model_utils",
    'pipeline',
    '__version__']


#def app():
#    print("¡basic_ML!")