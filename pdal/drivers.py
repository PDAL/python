import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, ClassVar, FrozenSet, Mapping, Optional, Sequence, Type

from .pipeline import Filter, Reader, Stage, Writer

StreamableTypes: FrozenSet


@dataclass
class Option:
    name: str
    description: str
    default: Optional[str] = None

    def __repr__(self) -> str:
        if self.default is not None:
            return f"{self.name}={self.default!r}: {self.description}"
        else:
            return f"{self.name}: {self.description}"


@dataclass
class Driver:
    name: str
    short_name: str = field(init=False)
    type: Type[Stage] = field(init=False)
    description: str
    options: Sequence[Option]

    def __post_init__(self) -> None:
        prefix, _, suffix = self.name.partition(".")
        self.type = self._prefix_to_type[prefix]
        self.short_name = suffix

    @property
    def factory(self) -> Callable[..., Stage]:
        if self.options and self.options[0].name == "filename":
            factory = lambda filename, **kwargs: self.type(
                filename=filename, type=self.name, **kwargs
            )
        else:
            factory = lambda **kwargs: self.type(type=self.name, **kwargs)
        factory.__name__ = self.short_name
        factory.__qualname__ = f"{self.type.__name__}.{self.short_name}"
        factory.__module__ = self.type.__module__
        factory.__doc__ = self.description
        if self.options:
            factory.__doc__ += "\n\n"
            factory.__doc__ += "\n".join(map(repr, self.options))
        return factory

    _prefix_to_type: ClassVar[Mapping[str, Type[Stage]]] = {
        "readers": Reader,
        "filters": Filter,
        "writers": Writer,
    }


def inject_pdal_drivers() -> None:
    drivers = json.loads(
        subprocess.run(["pdal", "--drivers", "--showjson"], capture_output=True).stdout
    )
    options = dict(
        json.loads(
            subprocess.run(
                ["pdal", "--options", "all", "--showjson"], capture_output=True
            ).stdout
        )
    )
    streamable = []
    for d in drivers:
        name = d["name"]
        d_options = [Option(**option_dict) for option_dict in (options.get(name) or ())]
        # move filename option first
        try:
            i = next(i for i, opt in enumerate(d_options) if opt.name == "filename")
            d_options.insert(0, d_options.pop(i))
        except StopIteration:
            pass
        driver = Driver(name, d["description"], d_options)
        setattr(driver.type, driver.short_name, staticmethod(driver.factory))
        if d["streamable"]:
            streamable.append(driver.name)
    global StreamableTypes
    StreamableTypes = frozenset(streamable)
