import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping, Optional, Sequence, Type

from .pipeline import Filter, Reader, Stage, Writer


@dataclass
class Option:
    name: str
    description: str
    default: Optional[str]

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
    options: Optional[Sequence[Option]]

    def __post_init__(self) -> None:
        prefix, _, suffix = self.name.partition(".")
        self.type = self._prefix_to_type[prefix]
        self.short_name = suffix

    @property
    def factory(self) -> Callable[..., Stage]:
        factory: Callable[..., Stage]
        if self.type is Reader:
            factory = lambda filename, **kwargs: Reader(
                filename, type=self.name, **kwargs
            )
        elif self.type is Writer:
            factory = lambda filename=None, **kwargs: Writer(
                filename, type=self.name, **kwargs
            )
        else:
            factory = lambda **kwargs: Filter(type=self.name, **kwargs)
        factory.__name__ = self.short_name
        factory.__qualname__ = f"{self.type.__name__}.{self.short_name}"
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
    for d in drivers:
        name = d["name"]
        d_options = options.get(name)
        if d_options is not None:
            d_options = [
                Option(o["name"], o["description"], o.get("default")) for o in d_options
            ]
            # move filename option first
            try:
                i = next(i for i, opt in enumerate(d_options) if opt.name == "filename")
                d_options.insert(0, d_options.pop(i))
            except StopIteration:
                pass
        driver = Driver(name, d["description"], d_options)
        setattr(driver.type, driver.short_name, staticmethod(driver.factory))