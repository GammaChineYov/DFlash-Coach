"""Shared annotations for the JSON-based portable interfaces."""

from os import PathLike as OSPathLike
from typing import Any, TypeAlias

PathLike: TypeAlias = str | OSPathLike[str]
Record: TypeAlias = dict[str, Any]
Config: TypeAlias = dict[str, Any]

SourceKey: TypeAlias = str | tuple[str, str]
