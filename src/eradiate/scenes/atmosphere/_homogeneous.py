import attr
import pint
import pinttr

from ._core import Atmosphere, atmosphere_factory
from ..core import KernelDict
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ..spectra import AirScatteringCoefficientSpectrum, Spectrum, spectrum_factory
from ..._util import onedict_value
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.transform import map_cube
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import has_quantity


@atmosphere_factory.register(type_id="homogeneous")
@parse_docs
@attr.s()
class HomogeneousAtmosphere(Atmosphere):
    """
    Homogeneous atmosphere scene element [``homogeneous``].

    This class builds an atmosphere consisting of a homogeneous medium with
    customisable collision coefficients and phase function, attached to a
    cuboid shape.
    """

    _bottom: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's bottom altitude.\n\nUnit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="quantity or float",
        default="0 km",
    )

    _top: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(10.0, ureg.km),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere's top altitude.\n\nUnit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="10 km.",
    )

    @_bottom.validator
    @_top.validator
    def _validate_bottom_and_top(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    sigma_s: Spectrum = documented(
        attr.ib(
            factory=AirScatteringCoefficientSpectrum,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere scattering coefficient value.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum` or float",
        default=":class:`AirScatteringCoefficient() <.AirScatteringCoefficient>`",
    )

    sigma_a: Spectrum = documented(
        attr.ib(
            default=0.0,
            converter=spectrum_factory.converter("collision_coefficient"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("collision_coefficient"),
            ],
        ),
        doc="Atmosphere absorption coefficient value. Defaults disable "
        "absorption.\n"
        "\n"
        "Can be initialised with a dictionary processed by "
        ":data:`~eradiate.scenes.spectra.spectrum_factory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.0 ucc[collision_coefficient]",
    )

    phase: PhaseFunction = documented(
        attr.ib(
            factory=lambda: RayleighPhaseFunction(),
            converter=phase_function_factory.convert,
            validator=attr.validators.instance_of(PhaseFunction),
        )
    )

    def __attrs_post_init__(self) -> None:
        self.update()

    def update(self) -> None:
        self.phase.id = self.id_phase

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        return self._bottom

    @property
    def top(self) -> pint.Quantity:
        return self._top

    # --------------------------------------------------------------------------
    #                           Evaluation methods
    # --------------------------------------------------------------------------

    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        if self.width is AUTO:
            spectral_ctx = ctx.spectral_ctx
            return 10.0 / self.eval_sigma_s(spectral_ctx)
        else:
            return self.width

    def eval_albedo(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return albedo.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Albedo.
        """
        return self.eval_sigma_s(spectral_ctx) / (
            self.eval_sigma_s(spectral_ctx) + self.eval_sigma_a(spectral_ctx)
        )

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return absorption coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Absorption coefficient.
        """
        return self.sigma_a.eval(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return scattering coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Scattering coefficient.
        """
        return self.sigma_s.eval(spectral_ctx)

    def eval_sigma_t(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Return extinction coefficient.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        quantity
            Extinction coefficient.
        """
        return self.eval_sigma_a(spectral_ctx) + self.eval_sigma_s(spectral_ctx)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        return self.phase.kernel_dict(ctx=ctx)

    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
        if ctx.ref:
            phase = {"type": "ref", "id": self.phase.id}
        else:
            phase = onedict_value(self.kernel_phase(ctx=ctx))

        return KernelDict(
            {
                self.id_medium: {
                    "type": "homogeneous",
                    "phase": phase,
                    "sigma_t": self.eval_sigma_t(ctx.spectral_ctx).m_as(
                        uck.get("collision_coefficient")
                    ),
                    "albedo": self.eval_albedo(ctx.spectral_ctx).m_as(
                        uck.get("albedo")
                    ),
                }
            }
        )

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        if ctx.ref:
            medium = {"type": "ref", "id": self.id_medium}
        else:
            medium = self.kernel_media(ctx=ctx)[self.id_medium]

        length_units = uck.get("length")
        width = self.kernel_width(ctx=ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        offset = self.kernel_offset(ctx=ctx).m_as(length_units)
        trafo = map_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom - offset,
            zmax=top,
        )

        return KernelDict(
            {
                f"shape_{self.id}": {
                    "type": "cube",
                    "to_world": trafo,
                    "bsdf": {"type": "null"},
                    "interior": medium,
                }
            }
        )