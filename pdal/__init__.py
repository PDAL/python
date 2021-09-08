from .libpdalpython import (
    getDebugInformation,
    getDimensions,
    getPluginInstallPath,
    getSha1,
    getVersionMajor,
    getVersionMinor,
    getVersionPatch,
    getVersionString,
)
from .pipeline import Pipeline


__version__ = "2.4.2"

dimensions = getDimensions()
del getDimensions


class Info(object):
    version = getVersionString()
    major = getVersionMajor()
    minor = getVersionMinor()
    patch = getVersionPatch()
    debug = getDebugInformation()
    sha1 = getSha1()
    plugin = getPluginInstallPath()


info = Info()
