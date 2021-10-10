from __future__ import annotations

import glob
import json
import subprocess
from typing import Any, Container, Dict, Iterator, List, Optional, Sequence, Union, cast

import numpy as np

from . import libpdalpython

_PDAL_DRIVERS = json.loads(
    subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
)


class Pipeline(libpdalpython.Pipeline):
    def __init__(
        self,
        spec: Union[None, str, Sequence[Stage]] = None,
        arrays: Sequence[np.ndarray] = (),
    ):
        self._stages: List[Stage] = []
        if spec:
            stages = _parse_stages(spec) if isinstance(spec, str) else spec
            for stage in stages:
                self |= stage
        if arrays:
            self.inputs = arrays

    @property
    def stages(self) -> List[Stage]:
        return list(self._stages)

    def __ior__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        if isinstance(other, Stage):
            self._stages.append(other)
        elif isinstance(other, Pipeline):
            if self._stages and other._num_inputs:
                raise ValueError(
                    "A pipeline with inputs cannot follow another pipeline"
                )
            self._stages.extend(other._stages)
        else:
            raise TypeError(f"Expected Stage or Pipeline, not {other}")
        self._delete_executor()
        return self

    def __or__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        new = self.__copy__()
        new |= other
        return new

    def __copy__(self) -> Pipeline:
        clone = cast(Pipeline, super().__copy__())
        clone |= self
        return clone

    @property
    def _json(self) -> str:
        options_list = []
        stage2tag: Dict[Stage, str] = {}
        for stage in self._stages:
            stage2tag[stage] = stage.tag or _generate_tag(stage, stage2tag.values())
            options = stage.options
            options["tag"] = stage2tag[stage]
            inputs = _get_input_tags(stage, stage2tag)
            if inputs:
                options["inputs"] = inputs
            options_list.append(options)
        return json.dumps(options_list)


class Stage:
    def __init_subclass__(cls, type_prefix: str) -> None:
        for driver in _PDAL_DRIVERS:
            name = driver["name"]
            prefix, _, suffix = name.partition(".")
            if prefix == type_prefix:
                cls._add_constructor(name, suffix, driver["description"])

    def __init__(self, **options: Any):
        self._options = options

    @property
    def type(self) -> str:
        return cast(str, self._options["type"])

    @property
    def tag(self) -> Optional[str]:
        return self._options.get("tag")

    @property
    def inputs(self) -> List[Union[Stage, str]]:
        inputs = self._options.get("inputs", ())
        return [inputs] if isinstance(inputs, (Stage, str)) else list(inputs)

    @property
    def options(self) -> Dict[str, Any]:
        return dict(self._options)

    def pipeline(self, *arrays: np.ndarray) -> Pipeline:
        return Pipeline((self,), arrays=arrays)

    def __or__(self, other: Union[Stage, Pipeline]) -> Pipeline:
        return Pipeline((self, other))

    @classmethod
    def _add_constructor(cls, type: str, name: str, description: str) -> None:
        constructor = lambda cls, *args, **kwargs: cls(*args, **kwargs, type=type)
        constructor.__name__ = name
        constructor.__qualname__ = f"{cls.__name__}.{name}"
        constructor.__doc__ = description
        setattr(cls, name, classmethod(constructor))


class Reader(Stage, type_prefix="readers"):
    def __init__(self, filename: str, **options: Any):
        super().__init__(filename=filename, **options)

    @property
    def type(self) -> str:
        try:
            return super().type
        except KeyError:
            filename = self._options["filename"]
            return cast(str, libpdalpython.infer_reader_driver(filename))


class Filter(Stage, type_prefix="filters"):
    def __init__(self, type: str, **options: Any):
        super().__init__(type=type, **options)


class Writer(Stage, type_prefix="writers"):
    def __init__(self, filename: Optional[str] = None, **options: Any):
        super().__init__(filename=filename, **options)

    @property
    def type(self) -> str:
        try:
            return super().type
        except KeyError:
            filename = self._options["filename"]
            return cast(str, libpdalpython.infer_writer_driver(filename))


def _parse_stages(text: str) -> Iterator[Stage]:
    json_stages = json.loads(text)
    if isinstance(json_stages, dict):
        json_stages = json_stages.get("pipeline")
    if not isinstance(json_stages, list):
        raise ValueError("root element is not a pipeline")

    last = len(json_stages) - 1
    for i, options in enumerate(json_stages):
        if not isinstance(options, dict):
            if isinstance(options, str):
                options = {"filename": options}
            else:
                raise ValueError("A stage element must be string or dict")

        stage_type = options.get("type")
        if stage_type:
            is_reader = stage_type.startswith("readers.")
        else:
            # The type is inferred from a filename as a reader if it's not
            # the last stage or if there's only one.
            is_reader = i == 0 or i != last

        if is_reader:
            paths = glob.glob(options.get("filename", ""))
            if paths:
                del options["filename"]
                for path in paths:
                    yield Reader(filename=path, **options)
            else:
                yield Reader(**options)
        elif not stage_type or stage_type.startswith("writers."):
            yield Writer(**options)
        else:
            yield Filter(**options)


def _generate_tag(stage: Stage, tags: Container[str]) -> str:
    tag_prefix = stage.type.replace(".", "_")
    i = 1
    while True:
        tag = tag_prefix + str(i)
        if tag not in tags:
            return tag
        i += 1


def _get_input_tags(stage: Stage, stage2tag: Dict[Stage, str]) -> List[str]:
    tags = []
    for input in stage.inputs:
        if isinstance(input, Stage):
            try:
                tags.append(stage2tag[input])
            except KeyError:
                raise RuntimeError(
                    f"Invalid pipeline: Undefined stage " f"{input.tag or input.type!r}"
                )
        else:
            tags.append(input)
    return tags
