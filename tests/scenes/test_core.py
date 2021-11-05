import importlib

import mitsuba
import numpy as np
import pint
import pytest

from eradiate import unit_registry as ureg
from eradiate.exceptions import KernelVariantError
from eradiate.scenes.core import BoundingBox, KernelDict


def test_kernel_dict_construct():
    # Object creation is possible only if a variant is set
    importlib.reload(
        mitsuba
    )  # Required to ensure that any variant set by another test is unset
    with pytest.raises(KernelVariantError):
        KernelDict()
    mitsuba.set_variant("scalar_mono")

    # variant attribute is set properly
    kernel_dict = KernelDict({})
    assert kernel_dict.variant == "scalar_mono"


def test_kernel_dict_check(mode_mono):
    # Check method raises upon missing scene type
    kernel_dict = KernelDict({})
    with pytest.raises(ValueError):
        kernel_dict.check()

    # Check method raises if dict and set variants are incompatible
    mitsuba.set_variant("scalar_mono_double")
    with pytest.raises(KernelVariantError):
        kernel_dict.check()


def test_kernel_dict_load(mode_mono):
    # Load method returns a kernel object
    from mitsuba.render import Scene, Shape

    kernel_dict = KernelDict({"type": "scene", "shape": {"type": "sphere"}})
    assert isinstance(kernel_dict.load(), Scene)

    # Also works if "type" is missing
    kernel_dict = KernelDict({"shape": {"type": "sphere"}})
    assert isinstance(kernel_dict.load(strip=False), Scene)

    # Setting strip to True instantiates a Shape directly...
    kernel_dict = KernelDict({"shape": {"type": "sphere"}})
    assert isinstance(kernel_dict.load(strip=True), Shape)

    # ... but not if the dict has two entries
    kernel_dict = KernelDict(
        {
            "shape_1": {"type": "sphere"},
            "shape_2": {"type": "sphere"},
        }
    )
    assert isinstance(kernel_dict.load(strip=True), Scene)


def test_kernel_dict_post_load(mode_mono):
    from mitsuba.python.util import traverse

    kernel_dict = KernelDict(
        data={
            "type": "directional",
            "irradiance": {
                "type": "irregular",
                "wavelengths": "400, 500",
                "values": "1, 1",
            },
        },
        post_load={
            "irradiance.wavelengths": np.array([400.0, 500.0, 600.0]),
            "irradiance.values": np.array([0.0, 1.0, 2.0]),
        },
    )

    # Without post-load update, buffers are initialised as in data
    obj = kernel_dict.load(post_load_update=False)
    params = traverse(obj)
    assert params["irradiance.wavelengths"] == np.array([400.0, 500.0])
    assert params["irradiance.values"] == np.array([1.0, 1.0])

    # Without post-load update, buffers are initialised as in post_load
    obj = kernel_dict.load(post_load_update=True)
    params = traverse(obj)
    assert params["irradiance.wavelengths"] == np.array([400.0, 500.0, 600.0])
    assert params["irradiance.values"] == np.array([0.0, 1.0, 2.0])


def test_kernel_dict_duplicate_id():
    from eradiate.contexts import KernelDictContext
    from eradiate.scenes.illumination import illumination_factory

    # Upon trying to add a scene element, whose ID is already present
    # in the KernelDict, a Warning must be issued.
    with pytest.warns(Warning):
        kd = KernelDict(
            {
                "testmeasure": {
                    "type": "directional",
                    "id": "testmeasure",
                    "irradiance": {
                        "type": "interpolated",
                        "wavelengths": [400, 500],
                        "values": [1, 1],
                    },
                }
            }
        )

        kd.add(
            illumination_factory.convert(
                {
                    "type": "directional",
                    "id": "testmeasure",
                    "irradiance": {
                        "type": "interpolated",
                        "wavelengths": [400, 500],
                        "values": [2, 2],
                    },
                }
            ),
            ctx=KernelDictContext(),
        )


def test_bbox():
    # Instantiation with correctly ordered unitless values works
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])
    assert bbox.units == ureg.m

    # Instantiation with correctly ordered unit-attached values works
    bbox = BoundingBox([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m)
    assert bbox.min.units == ureg.m

    # Extents are correctly computed
    assert np.allclose([1, 2, 3] * ureg.m, BoundingBox([0, 0, 0], [1, 2, 3]).extents)

    # Extent shapes must be compatible
    with pytest.raises(ValueError):
        BoundingBox([0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        BoundingBox([0, 0, 0], [[1, 1, 1]])

    # Unit mismatch raises
    with pytest.raises(pint.DimensionalityError):
        BoundingBox([0, 0] * ureg.m, [1, 1] * ureg.s)


def test_bbox_convert():
    bbox_ref = BoundingBox([0, 0, 0], [1, 1, 1])

    bbox_convert = BoundingBox.convert([[0, 0, 0], [1, 1, 1]])
    assert np.allclose(bbox_ref.min, bbox_convert.min)
    assert np.allclose(bbox_ref.max, bbox_convert.max)

    bbox_convert = BoundingBox.convert(np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.allclose(bbox_ref.min, bbox_convert.min)
    assert np.allclose(bbox_ref.max, bbox_convert.max)

    bbox_ref = BoundingBox([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m)
    bbox_convert = BoundingBox.convert([[0, 0, 0], [1, 1, 1]] * ureg.m)
    assert np.allclose(bbox_ref.min, bbox_convert.min)
    assert np.allclose(bbox_ref.max, bbox_convert.max)