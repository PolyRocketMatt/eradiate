import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, MonoSpectralContext, SpectralContext
from eradiate.scenes.atmosphere._particle_dist import UniformParticleDistribution
from eradiate.scenes.atmosphere._particle_layer import ParticleLayer
from eradiate.scenes.measure._core import CKDMeasureSpectralConfig
from eradiate.units import symbol, to_quantity


def to_dataset(albedo, sigma_t, phase, mu, w):
    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "w",
                sigma_t.magnitude,
                dict(
                    standard_name="air_volume_extinction_coefficient",
                    units=symbol(sigma_t.units),
                ),
            ),
            "albedo": (
                "w",
                albedo.magnitude,
                dict(
                    standard_name="single_scattering_albedo", units=symbol(albedo.units)
                ),
            ),
            "phase": (
                ("w", "mu", "i", "j"),
                phase.magnitude,
                dict(
                    standard_name="scattering_phase_matrix", units=symbol(phase.units)
                ),
            ),
        },
        coords={
            "w": ("w", w.magnitude, dict(units=symbol(w.units))),
            "mu": (
                "mu",
                mu.magnitude,
                dict(standard_name="scattering_angle_cosine", units=f"{mu.units:~}"),
            ),
            "i": ("i", [0]),
            "j": ("j", [0]),
        },
    )


@pytest.fixture
def absorbing_only(tmpdir):
    """Absorbing only particles radiative properties data set path fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.zeros_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    radprops = to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)
    path = tmpdir / "absorbing_particles.nc"
    radprops.to_netcdf(path)
    return path


@pytest.fixture
def scattering_only(tmpdir):
    """Scattering only particles radiative properties data set path fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = np.ones_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    radprops = to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)
    path = tmpdir / "scattering_particles.nc"
    radprops.to_netcdf(path)
    return path


@pytest.fixture
def test_particles_dataset(tmpdir):
    """Particles radiative properties data set path fixture."""
    mu = np.linspace(-1.0, 1.0) * ureg.dimensionless
    w = np.linspace(279.0, 2401.0) * ureg.nm
    arrays = [np.ones_like(mu) / ureg.steradian for _ in w]
    phase = np.stack(arrays, axis=0).reshape(w.size, mu.size, 1, 1)
    albedo = 0.8 * np.ones_like(w) * ureg.dimensionless
    sigma_t = np.ones_like(w) / ureg.km
    radprops = to_dataset(albedo=albedo, sigma_t=sigma_t, phase=phase, mu=mu, w=w)
    path = tmpdir / "test_particles_dataset.nc"
    radprops.to_netcdf(path)
    return path


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono_absorbing_only(
    mode_mono, tmpdir, absorbing_only, wavelength
):
    """eval methods return expected values for an absorbing-only layer."""
    layer = ParticleLayer(dataset=absorbing_only)
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.all(layer.eval_sigma_s(spectral_ctx).magnitude == 0.0)
    assert np.all(layer.eval_sigma_a(spectral_ctx).magnitude > 0.0)
    assert np.all(layer.eval_albedo(spectral_ctx).magnitude == 0.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx).magnitude > 0.0


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono_scattering_only(
    mode_mono, tmpdir, scattering_only, wavelength
):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(dataset=scattering_only)
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.all(layer.eval_sigma_s(spectral_ctx).magnitude > 0.0)
    assert np.all(layer.eval_sigma_a(spectral_ctx).magnitude == 0.0)
    assert np.all(layer.eval_albedo(spectral_ctx).magnitude == 1.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx).magnitude > 0.0


@pytest.mark.parametrize("wavelength", [280.0, 550.0, 1600.0, 2400.0])
def test_particle_layer_eval_mono(
    mode_mono, tmpdir, test_particles_dataset, wavelength
):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(
        dataset=test_particles_dataset,
        n_layers=1,
        tau_550=1.0,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
    )
    spectral_ctx = MonoSpectralContext(wavelength=wavelength)
    assert np.isclose(layer.eval_sigma_t(spectral_ctx), 1.0 / ureg.km)
    assert np.isclose(layer.eval_sigma_s(spectral_ctx), 0.8 / ureg.km)
    assert np.isclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.isclose(layer.eval_albedo(spectral_ctx).magnitude, 0.8)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx) == 12.5 * ureg.km


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd_absorbing_only(mode_ckd, tmpdir, absorbing_only, bins):
    """eval methods return expected values for an absorbing-only layer."""
    layer = ParticleLayer(dataset=absorbing_only)
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.all(layer.eval_sigma_s(spectral_ctx).magnitude == 0.0)
    assert np.all(layer.eval_sigma_a(spectral_ctx).magnitude > 0.0)
    assert np.all(layer.eval_albedo(spectral_ctx).magnitude == 0.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx).magnitude > 0.0


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd_scattering_only(
    mode_ckd, tmpdir, scattering_only, bins
):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(dataset=scattering_only)
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.all(layer.eval_sigma_s(spectral_ctx).magnitude > 0.0)
    assert np.all(layer.eval_sigma_a(spectral_ctx).magnitude == 0.0)
    assert np.all(layer.eval_albedo(spectral_ctx).magnitude == 1.0)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx).magnitude > 0.0


@pytest.mark.parametrize("bins", ["280", "550", "1600", "2400"])
def test_particle_layer_eval_ckd(mode_ckd, tmpdir, test_particles_dataset, bins):
    """eval methods return expected values for a scattering-only layer."""
    layer = ParticleLayer(
        dataset=test_particles_dataset,
        n_layers=1,
        tau_550=1.0,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
    )
    spectral_config = CKDMeasureSpectralConfig(bin_set="10nm", bins=bins)
    spectral_ctx = spectral_config.spectral_ctxs()[0]
    assert np.isclose(layer.eval_sigma_t(spectral_ctx), 1.0 / ureg.km)
    assert np.isclose(layer.eval_sigma_s(spectral_ctx), 0.8 / ureg.km)
    assert np.isclose(layer.eval_sigma_a(spectral_ctx), 0.2 / ureg.km)
    assert np.isclose(layer.eval_albedo(spectral_ctx).magnitude, 0.8)

    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    assert layer.eval_width(ctx) == 12.5 * ureg.km


def test_particle_layer_construct_basic():
    """Construction succeeds with basic parameters."""
    assert ParticleLayer(n_layers=9)


def test_particle_layer_scale(modes_all_single):
    """Scale parameter propagates to kernel dict and latter can be loaded."""
    ctx = KernelDictContext()
    d = ParticleLayer(scale=2.0).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert d.load()


def test_particle_layer_construct_attrs():
    """Assigns parameters to expected values."""
    bottom = ureg.Quantity(1.2, "km")
    top = ureg.Quantity(1.8, "km")
    tau_550 = ureg.Quantity(0.3, "dimensionless")
    layer = ParticleLayer(
        bottom=bottom,
        top=top,
        distribution=UniformParticleDistribution(),
        tau_550=tau_550,
        n_layers=9,
        dataset="tests/radprops/rtmom_aeronet_desert.nc",
    )
    assert layer.bottom == bottom
    assert layer.top == top
    assert isinstance(layer.distribution, UniformParticleDistribution)
    assert layer.tau_550 == tau_550
    assert layer.n_layers == 9
    assert layer.dataset == path_resolver.resolve(
        "tests/radprops/rtmom_aeronet_desert.nc"
    )


def test_particle_layer_altitude_units():
    """Accept different units for bottom and top altitudes."""
    assert ParticleLayer(bottom=1.0 * ureg.km, top=2000.0 * ureg.m)


def test_particle_layer_invalid_bottom_top():
    """Raises when 'bottom' is larger that 'top'."""
    with pytest.raises(ValueError):
        ParticleLayer(top=1.2 * ureg.km, bottom=1.8 * ureg.km)


def test_particle_layer_invalid_tau_550():
    """Raises when 'tau_550' is invalid."""
    with pytest.raises(ValueError):
        ParticleLayer(
            bottom=1.2 * ureg.km,
            top=1.8 * ureg.km,
            tau_550=-0.1 * ureg.dimensionless,
        )


def test_particle_layer_kernel_phase(modes_all_single):
    """Dictionary key is set to appropriate value."""
    atmosphere = ParticleLayer(n_layers=9)
    ctx = KernelDictContext()
    kernel_phase = atmosphere.kernel_phase(ctx)
    assert set(kernel_phase.data.keys()) == {f"phase_{atmosphere.id}"}


def test_particle_layer_kernel_dict(modes_all_single):
    """Kernel dictionary can be loaded"""
    particle_layer = ParticleLayer(n_layers=9)
    ctx = KernelDictContext()
    assert particle_layer.kernel_dict(ctx).load()


@pytest.fixture
def test_dataset():
    """Test dataset path fixture."""
    return path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc")


def test_particle_layer_eval_radprops(modes_all_single, test_dataset):
    """Method 'eval_radprops' returns dataset with expected datavars and coords."""
    layer = ParticleLayer(dataset=test_dataset)
    spectral_ctx = SpectralContext.new()
    ds = layer.eval_radprops(spectral_ctx)
    expected_data_vars = ["sigma_t", "albedo"]
    expected_coords = ["z_layer"]
    assert all([coord in ds.coords for coord in expected_coords]) and all(
        [var in ds.data_vars for var in expected_data_vars]
    )


@pytest.mark.parametrize("tau_550", [0.1, 0.5, 1.0, 5.0])
def test_particle_layer_eval_sigma_t_mono(mode_mono, tau_550, test_dataset):
    r"""
    Spectral dependency of extinction is accounted for.

    If :math:`\sigma_t(\lambda)` denotes the extinction coefficient at the
    wavelength :math:`\lambda`, then the optical thickness of a uniform
    particle layer is :math:`\tau(\lambda) = \sigma_t(\lambda) \, \Delta z`
    where :math:`\Delta z` is the layer's thickness.
    It follows that:

    .. math::

       \frac{\tau(\lambda)}{\tau(550\, \mathrm{nm})} =
       \frac{\sigma(\lambda)}{\sigma(550\, \mathrm{nm})}

    which is what we assert in this test.
    """
    wavelengths = np.linspace(500.0, 1500.0, 1001) * ureg.nm
    tau_550 = tau_550 * ureg.dimensionless

    # tau_550 = 1.0 * ureg.dimensionless
    layer = ParticleLayer(
        dataset=test_dataset,
        bottom=0.0 * ureg.km,
        top=1.0 * ureg.km,
        distribution={"type": "uniform"},
        n_layers=1,
        tau_550=tau_550,
    )

    # layer optical thickness @ current wavelengths
    tau = layer.eval_sigma_t_mono(wavelengths) * layer.height

    # data set extinction @ running wavelength and 550 nm
    with xr.open_dataset(test_dataset) as ds:
        w_units = ureg(ds.w.attrs["units"])
        sigma_t = to_quantity(ds.sigma_t.interp(w=wavelengths.m_as(w_units)))
        sigma_t_550 = to_quantity(ds.sigma_t.interp(w=(550.0 * ureg.nm).m_as(w_units)))

    # the spectral dependence of the optical thickness and extinction coefficient
    # match, so the below ratios must match
    assert np.allclose(tau / tau_550, sigma_t / sigma_t_550)