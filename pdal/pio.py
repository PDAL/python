"""
This module provides a python-syntax interface for constructing and executing pdal-python json
pipelines.  The API is not explicitly defined but stage names are validated against the pdal executable's drivers when possible.

To construct pipeline stages, access the driver name from this module.  This will create
a callable function where driver parameters can be specified as keyword arguments.  For example:

>>> from pdal import pio
>>> las_reader = pio.readers.las(filename="test.las")

To construct a pipeline, sum stages together.

>>> pipeline = pio.readers.las(filename="test.las") + pio.writers.ply(filename="test.ply")

To execute a pipeline and return results, call `execute`.

>>> arr = pipeline.execute() # returns a numpy structured array

To access the pipelines as a dict (which may be dumped to json), call `spec`.

>>> json.dumps(pipeline.spec)

"""

import types
import json
import subprocess
from functools import partial
from collections import defaultdict
from itertools import chain
import copy
import warnings

import pdal

try:
    PDAL_DRIVERS_JSON = subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
    PDAL_DRIVERS = json.loads(PDAL_DRIVERS_JSON)
    _PDAL_VALIDATE = True
except:
    PDAL_DRIVERS = []
    _PDAL_VALIDATE = False

DEFAULT_STAGE_PARAMS = defaultdict(dict)
DEFAULT_STAGE_PARAMS.update({
# TODO: add stage specific default configurations
})


class StageSpec(object):
    def __init__(self, prefix, **kwargs):
        self.prefix = prefix
        self.key = ".".join([self.prefix, kwargs.get("type", "")])
        self.spec = DEFAULT_STAGE_PARAMS[self.key].copy()
        self.spec.update(kwargs)
        self.spec["type"] = self.key
        # NOTE: special case to support reading files without passing an explicit reader
        if (self.prefix in ["readers", "writers"]) and kwargs.get("type") == "auto":
            del self.spec["type"]

    @property
    def pipeline(self):
        """
        Promote this stage to  a `pdal.pio.PipelineSpec` with one `pdal.pio.StageSpec`
        and return it.
        """
        output = PipelineSpec()
        output.add_stage(self)
        return output

    def __getattr__(self, name):
        if _PDAL_VALIDATE and (name not in dir(self)):
            raise AttributeError(f"'{self.prefix}.{name}' is an invalid or unsupported PDAL stage")
        return partial(self.__class__, self.prefix, type=name)

    def __str__(self):
        return json.dumps(self.spec, indent=4)

    def __add__(self, other):
        return self.pipeline + other

    def __dir__(self):
        extra_keys = [e["name"][len(self.key):] for e in PDAL_DRIVERS if e["name"].startswith(self.key)] + ["auto"]
        return super().__dir__() + [e for e in extra_keys if len(e) > 0]

    def execute(self):
        return self.pipeline.execute()


readers = StageSpec("readers")
filters = StageSpec("filters")
writers = StageSpec("writers")


class PipelineSpec(object):
    stages = []

    def __init__(self, other=None):
        if other is not None:
            self.stages = copy.copy(other.stages)

    @property
    def spec(self):
        """
        Return a `dict` containing the pdal pipeline suitable for dumping to json
        """
        return {
            "pipeline": [stage.spec for stage in self.stages]
        }

    def add_stage(self, stage):
        """
        Add a StageSpec to the end of this pipeline, and return the updated result.
        """
        assert isinstance(stage, StageSpec), "Expected StageSpec"

        self.stages.append(stage)
        return self

    def __str__(self):
        return json.dumps(self.spec, indent=4)

    def __add__(self, stage_or_pipeline):
        assert isinstance(stage_or_pipeline, (StageSpec, PipelineSpec)), "Expected StageSpec or PipelineSpec"

        output = self.__class__(self)
        if isinstance(stage_or_pipeline, StageSpec):
            output.add_stage(stage_or_pipeline)
        elif isinstance(stage_or_pipeline, PipelineSpec):
            for stage in stage_or_pipeline.stages:
                output.add_stage(stage)
        return output

    def execute(self):
        """
        Shortcut to execute and return the results of the pipeline.
        """
        # TODO: do some validation before calling execute

        # TODO: some exception/error handling around pdal
        pipeline = pdal.Pipeline(json.dumps(self.spec))
        # pipeline.validate() # NOTE: disabling this because it causes segfaults in certain cases
        pipeline.execute()

        return pipeline.arrays[0] # NOTE: are there situation where arrays has multiple elements?
