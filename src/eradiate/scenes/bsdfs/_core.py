from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attr

from ..core import SceneElement
from ..._factory import Factory
from ...attrs import documented, get_doc

bsdf_factory = Factory()


@attr.s
class BSDF(SceneElement, ABC):
    """
    Abstract interface for all BSDF scene elements.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="bsdf",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default='"bsdf"',
    )