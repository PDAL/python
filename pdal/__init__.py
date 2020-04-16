__version__='2.3.2'

from .pipeline import Pipeline
from .array import Array
from .dimension import dimensions

from pdal.libpdalpython import getVersionString, getVersionMajor, getVersionMinor, getVersionPatch, getSha1, getDebugInformation, getPluginInstallPath

class Info(object):
    version = getVersionString()
    major = getVersionMajor()
    minor = getVersionMinor()
    patch = getVersionPatch()
    debug = getDebugInformation()
    sha1 = getSha1()
    plugin = getPluginInstallPath()

info = Info()
