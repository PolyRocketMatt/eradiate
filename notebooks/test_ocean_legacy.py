import pytest
import drjit as dr
import mitsuba as mi
import numpy as np

from ..tools import sample_eval_pdf_bsdf

_bsdf_dict = {
    'type': 'oceanic_legacy',
    'component': 0,
    'wavelength': 2.2,
    'wind_speed': 2,
    'wind_direction': 0,
    'chlorinity': 19,
    'pigmentation': 0.3,
    'shininess': 10.0,
}

@pytest.mark.slow
def test_consistency_oceanic(variants_vec_backends_once_rgb):
    """
    Test the consistency of the oceanic BSDF using the chi2 test.
    """
    sample_func, pdf_func = mi.chi2.BSDFAdapter("oceanic_legacy", _bsdf_dict)

    chi2 = mi.chi2.ChiSquareTest(
        domain=mi.chi2.SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
        ires=16,
        res=201
    )

    assert chi2.run()

def test_create_oceanic(variants_vec_backends_once_rgb):
    # Test constructor of oceanic BSDF
    brdf = mi.load_dict(_bsdf_dict)
    diff, gloss, comb = brdf.flags(0), brdf.flags(1), brdf.flags()

    # Obtain binary Mitsuba flags
    diff_flag = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
    gloss_flag = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
    comb_flag = diff_flag | gloss_flag

    assert isinstance(brdf, mi.BSDF)    
    assert diff == diff_flag
    assert gloss == gloss_flag
    assert comb == comb_flag