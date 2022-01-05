import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.experiments._rami4atm import Rami4ATMExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.biosphere import DiscreteCanopy
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.surface import CentralPatchSurface, LambertianSurface


def test_rami4atm_experiment_construct_default(mode_mono):
    """
    Rami4ATMExperiment initialises with default params in all modes
    """
    assert Rami4ATMExperiment()


def test_rami4atm_experiment_construct_measures(mode_mono):
    """
    A variety of measure specifications are acceptable
    """

    # Init with a single measure (not wrapped in a sequence)
    assert Rami4ATMExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert Rami4ATMExperiment(measures=[{"type": "distant"}])
    # -- Not wrapped in a sequence
    assert Rami4ATMExperiment(measures={"type": "distant"})


@pytest.mark.parametrize("padding", (0, 1))
def test_rami4atm_experiment_construct_normalize_measures(mode_mono, padding):

    # When canopy is not None, measure target matches canopy unit cell
    exp = Rami4ATMExperiment(
        atmosphere=None,
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=MultiDistantMeasure(),
    )
    target = exp.measures[0].target
    canopy = exp.canopy
    assert np.isclose(target.xmin, -0.5 * canopy.size[0])
    assert np.isclose(target.xmax, 0.5 * canopy.size[0])
    assert np.isclose(target.ymin, -0.5 * canopy.size[1])
    assert np.isclose(target.ymax, 0.5 * canopy.size[1])
    assert np.isclose(target.z, canopy.size[2])

    # The measure target does not depend on the atmosphere
    exp = Rami4ATMExperiment(
        atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42.0, "km")),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=MultiDistantMeasure(),
    )
    target = exp.measures[0].target
    canopy = exp.canopy
    assert np.isclose(target.xmin, -0.5 * canopy.size[0])
    assert np.isclose(target.xmax, 0.5 * canopy.size[0])
    assert np.isclose(target.ymin, -0.5 * canopy.size[1])
    assert np.isclose(target.ymax, 0.5 * canopy.size[1])
    assert np.isclose(target.z, canopy.size[2])


@pytest.mark.parametrize("padding", (0, 1))
def test_ramiatm_experiment_kernel_dict(mode_mono, padding):
    from mitsuba.core import Point3f, ScalarTransform4f

    ctx = KernelDictContext()

    # Surface width is appropriately inherited from canopy, when no atmosphere is present
    s = Rami4ATMExperiment(
        atmosphere=None,
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
        measures=[
            {"type": "distant", "id": "distant_measure"},
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    kernel_scene = s.kernel_dict(ctx)
    assert np.allclose(
        kernel_scene["surface"]["to_world"].transform_affine(Point3f(1, -1, 0)),
        [5 * (2 * padding + 1), -5 * (2 * padding + 1), 0],
    )

    # -- Measures get no external medium assigned
    assert "medium" not in kernel_scene["distant_measure"]
    assert "medium" not in kernel_scene["radiancemeter"]

    # Surface width is appropriately inherited from atmosphere
    s = Rami4ATMExperiment(
        atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42.0, "km")),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=padding,
        ),
    )
    kernel_dict = s.kernel_dict(ctx)
    assert np.allclose(
        kernel_dict["surface"]["to_world"].matrix,
        ScalarTransform4f.scale([21000, 21000, 1]).matrix,
    )


@pytest.mark.slow
def test_ramiatm_experiment_surface_adjustment(mode_mono):
    """Create a Rami4ATM experiment and assert the central patch surface is created with the
    correct parameters, according to the canopy and atmosphere."""
    from mitsuba.core import ScalarTransform4f

    ctx = KernelDictContext()

    s = Rami4ATMExperiment(
        atmosphere=HomogeneousAtmosphere(width=ureg.Quantity(42.0, "km")),
        canopy=DiscreteCanopy.homogeneous(
            lai=3.0,
            leaf_radius=0.1 * ureg.m,
            l_horizontal=10.0 * ureg.m,
            l_vertical=2.0 * ureg.m,
            padding=0,
        ),
        surface=CentralPatchSurface(
            central_patch=LambertianSurface(), background_surface=LambertianSurface()
        ),
    )

    expected_trafo = ScalarTransform4f.scale(1400) * ScalarTransform4f.translate(
        (-0.499642857, -0.499642857, 0.0)
    )

    kernel_dict = s.kernel_dict(ctx=ctx)

    assert np.allclose(
        kernel_dict["bsdf_surface"]["weight"]["to_uv"].matrix, expected_trafo.matrix
    )


@pytest.mark.slow
def test_ramiatm_experiment_real_life(mode_mono):
    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = eradiate.path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )

    # Construct with typical parameters
    exp = Rami4ATMExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa1976",
                "absorption_data_sets": dict(us76_u86_4=test_absorption_data_set),
            },
        },
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {
                "type": "distant",
                "construct": "from_viewing_angles",
                "zeniths": np.arange(-60, 61, 5),
                "azimuths": 0.0,
                "id": "distant",
            },
            {"type": "radiancemeter", "origin": [1, 0, 0], "id": "radiancemeter"},
        ],
    )
    assert exp.kernel_dict(ctx=ctx).load() is not None

    # -- Distant measures get no external medium
    assert "medium" not in exp.kernel_dict(ctx=ctx)["distant"]

    # -- Radiancemeter inside the atmosphere must have a medium assigned
    assert exp.kernel_dict(ctx=ctx)["radiancemeter"]["medium"] == {
        "type": "ref",
        "id": "medium_atmosphere",
    }


@pytest.mark.slow
def test_ramiatm_experiment_run_detailed(mode_mono):
    """
    Test for correctness of the result dataset generated by Rami4ATMExperiment.
    Note: This test is outdated, most of its content should be transferred to
    tests for measure post-processing pipelines.
    """
    exp = Rami4ATMExperiment(
        measures=[
            {
                "id": "toa_brf",
                "type": "distant",
                "construct": "from_viewing_angles",
                "zeniths": np.arange(-60, 61, 5),
                "azimuths": 0.0,
            },
        ]
    )

    exp.run()

    results = exp.results["toa_brf"]

    # Post-processing creates expected variables ...
    assert set(results.data_vars) == {
        "irradiance",
        "brf",
        "brdf",
        "radiance",
        "spp",
        "srf",
    }

    # ... dimensions
    assert set(results["radiance"].dims) == {"sza", "saa", "x_index", "y_index", "w"}
    assert set(results["irradiance"].dims) == {"sza", "saa", "w"}

    # ... and other coordinates
    expected_coords = {"sza", "saa", "vza", "vaa", "x", "x_index", "y", "y_index", "w"}
    assert set(results["radiance"].coords) == expected_coords
    assert set(results["irradiance"].coords) == {"sza", "saa", "w"}

    # We just check that we record something as expected
    assert np.all(results["radiance"].data > 0.0)


def test_onedim_experiment_inconsistent_multiradiancemeter(mode_mono):
    # A MultiRadiancemeter measure must have all origins inside the atmosphere or none.
    # A mix of both will raise an error.

    ctx = KernelDictContext()

    # Construct with typical parameters
    test_absorption_data_set = eradiate.path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    exp = Rami4ATMExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "construct": "ussa1976",
                "absorption_data_sets": dict(us76_u86_4=test_absorption_data_set),
            },
        },
        canopy={
            "type": "discrete_canopy",
            "construct": "homogeneous",
            "lai": 3.0,
            "leaf_radius": 0.1 * ureg.m,
            "l_horizontal": 10.0 * ureg.m,
            "l_vertical": 2.0 * ureg.m,
        },
        illumination={"type": "directional", "zenith": 45.0},
        measures=[
            {
                "type": "multi_radiancemeter",
                "origins": [[1, 0, 0], [1000000, 0, 0]],
                "directions": [[0, 0, -1], [0, 0, -1]],
                "id": "multi_radiancemeter",
            },
        ],
    )
    with pytest.raises(ValueError):
        exp.kernel_dict(ctx=ctx)