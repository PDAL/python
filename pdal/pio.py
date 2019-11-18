import json
from functools import partial
from collections import defaultdict
from itertools import chain
import copy
import warnings

import pdal

DEFAULT_STAGE_PARAMS = defaultdict(dict)
DEFAULT_STAGE_PARAMS.update({
# TODO: add stage specific default configurations
})

class StageSpec(object):
    def __init__(self, prefix, **kwargs):
        self.prefix = prefix
        key = ".".join([self.prefix, kwargs.get("type", "")])
        self.spec = DEFAULT_STAGE_PARAMS[key].copy()
        self.spec.update(kwargs)
        self.spec["type"] = key
        # NOTE: special case to support reading files without passing an explicit reader
        if (self.prefix == "readers") and kwargs.get("type") == "auto":
            del self.spec["type"]

    @property
    def pipeline(self):
        output = PipelineSpec()
        output.add_stage(self)
        return output

    def __getattr__(self, name):
        return partial(self.__class__, self.prefix, type=name)

    def __str__(self):
        return json.dumps(self.spec, indent=4)

    def __add__(self, other):
        return self.pipeline + other

    def execute(self):
        return self.pipeline.execute()


readers = StageSpec("readers")
filters = StageSpec("filters")
writers = StageSpec("writers")


class PipelineSpec(object):
    readers = []
    filters = []
    writer = None

    def __init__(self, other=None):
        if other is not None:
            self.readers = copy.copy(other.readers)
            self.filters = copy.copy(other.filters)
            self.writer = copy.copy(other.writer)

    @property
    def spec(self):
        return {
            "pipeline": [stage.spec for stage in self.stages]
        }

    @property
    def stages(self):
        if self.writer is not None:
            return chain(self.readers, self.filters, [self.writer])
        else:
            return chain(self.readers, self.filters)

    def add_stage(self, stage):
        assert isinstance(stage, StageSpec), "Expected StageSpec"
        if stage.prefix == "writers":
            if self.writer is not None:
                warnings.warn("Currently assigned output stage will be replaced.")
            self.writer = stage
        elif stage.prefix == "readers":
            self.readers.append(stage)
        elif stage.prefix == "filters":
            self.filters.append(stage)
        else:
            warnings.warn("Unknown stage type.  Skipping.")

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
        # TODO: do some validation before calling execute

        # TODO: some exception/error handling around pdal
        pipeline = pdal.Pipeline(json.dumps(self.spec))
        # pipeline.validate() # NOTE: disabling this because it causes segfaults in certain cases
        pipeline.execute()

        return pipeline.arrays[0] # NOTE: are there situation where arrays has multiple elements?
