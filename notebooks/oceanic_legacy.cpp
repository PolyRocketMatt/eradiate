#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/core/random.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <drjit/dynamic.h>

NAMESPACE_BEGIN(mitsuba)

#define AZIMUTH_QUADRATURE_PTS 16
#define ZENITH_QUADRATURE_PTS 8

// Header content - THIS CAN BE MOVED IF NECESSARY
#ifndef OCEAN_PROPS
#define OCEAN_PROPS

template<typename Float, typename Spectrum>
class OceanProperties {
public:
    MI_IMPORT_TYPES()

    using FloatX = DynamicBuffer<Float>;
    using Value = dr::Array<ScalarFloat, 1>;

    /**
     * @brief Construct a new Ocean Properties object and initializes the data.
     * 
     * Initializes the data for the effective reflectance of whitecaps, the
     * complex index of refraction of water, the water scattering and attenuation
     * coefficients, and the molecular scattering coefficients. The data is taken
     * from various sources in the literature. The quadrature points and weights
     * for azimuth and zenith are also computed and transformed to the appropriate
     * domains.
     */
    OceanProperties() {
        // Effective reflectance of whitecaps (Whitlock et al. 1982)
        std::vector<ScalarFloat> wc_wavelengths = {     0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                                                        1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
                                                        2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1,
                                                        3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0 };
        std::vector<ScalarFloat> wc_data = {    0.220, 0.220, 0.220, 0.220, 0.220, 0.220, 0.215, 0.210, 0.200, 0.190,
                                                0.175, 0.155, 0.130, 0.080, 0.100, 0.105, 0.100, 0.080, 0.045, 0.055,
                                                0.065, 0.060, 0.055, 0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                                                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 };

        // Complex index of refraction of water (Hale & Querry 1973)
        std::vector<ScalarFloat> ior_wavelengths = {    0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.345 ,0.375, 0.400, 0.425 ,
                                                        0.445, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675,
                                                        0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925,
                                                        0.950, 0.975, 1.000, 1.200, 1.400, 1.600, 1.800, 2.000, 2.200, 2.400, 
                                                        2.600, 2.650, 2.700, 2.750, 2.800, 2.850, 2.900, 2.950, 3.000, 3.050,
                                                        3.100, 3.150, 3.200, 3.250, 3.300, 3.350, 3.400, 3.450, 3.500, 3.600,
                                                        3.700, 3.800, 3.900, 4.000 };
        std::vector<ScalarFloat> ior_real_data = {  1.369, 1.373, 1.362, 1.354, 1.349, 1.346, 1.343, 1.341, 1.339, 1.338,
                                                    1.337, 1.336, 1.335, 1.334, 1.333, 1.333, 1.332, 1.332, 1.331, 1.331,
                                                    1.331, 1.330, 1.330, 1.330, 1.329, 1.329, 1.329, 1.328, 1.328, 1.328, 
                                                    1.327, 1.327, 1.327, 1.324, 1.321, 1.317, 1.312, 1.306, 1.296, 1.279,
                                                    1.242, 1.219, 1.188, 1.157, 1.142, 1.149, 1.201, 1.292, 1.371, 1.426,
                                                    1.467, 1.483, 1.478, 1.467, 1.450, 1.432, 1.420, 1.410, 1.400, 1.385,
                                                    1.374, 1.364, 1.357, 1.351 };
        std::vector<ScalarFloat> ior_cplx_data = {  1.10e-07, 4.90e-08, 3.35e-08, 2.35e-08, 1.60e-08, 
                                                    1.08e-08, 6.50e-09, 3.50e-09, 1.86e-09, 1.30e-09, 
                                                    1.02e-09, 9.35e-10, 1.00e-09, 1.32e-09, 1.96e-09, 
                                                    3.60e-09, 1.09e-08, 1.39e-08, 1.64e-08, 2.23e-08, 
                                                    3.35e-08, 9.15e-08, 1.56e-07, 1.48e-07, 1.25e-07, 
                                                    1.82e-07, 2.93e-07, 3.91e-07, 4.86e-07, 1.06e-06, 
                                                    2.93e-06, 3.48e-06, 2.89e-06, 9.89e-06, 1.38e-04, 
                                                    8.55e-05, 1.15e-04, 1.10e-03, 2.89e-04, 9.56e-04, 
                                                    3.17e-03, 6.70e-03, 1.90e-02, 5.90e-02, 1.15e-01, 
                                                    1.85e-01, 2.68e-01, 2.98e-01, 2.72e-01, 2.40e-01, 
                                                    1.92e-01, 1.35e-01, 9.24e-02, 6.10e-02, 3.68e-02, 
                                                    2.61e-02, 1.95e-02, 1.32e-02, 9.40e-03, 5.15e-03, 
                                                    3.60e-03, 3.40e-03, 3.80e-03, 4.60e-03 };

        // Water scattering and attenuation coefficient data (Morel 1988)
        std::vector<ScalarFloat> attn_wavelengths = {   400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 
                                                        450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 
                                                        500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 
                                                        550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 
                                                        600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 
                                                        650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 
                                                        700 };
        std::vector<ScalarFloat> attn_k = {     0.0209, 0.0200, 0.0196, 0.0189, 0.0183, 0.0182, 0.0171, 0.0170, 0.0168, 0.0166,
                                                0.0168, 0.0170, 0.0173, 0.0174, 0.0175, 0.0184, 0.0194, 0.0203, 0.0217, 0.0240,
                                                0.0271, 0.0320, 0.0384, 0.0445, 0.0490, 0.0505, 0.0518, 0.0543, 0.0568, 0.0615,
                                                0.0640, 0.0640, 0.0717, 0.0762, 0.0807, 0.0940, 0.1070, 0.1280, 0.1570, 0.2000,
                                                0.2530, 0.2790, 0.2960, 0.3030, 0.3100, 0.3150, 0.3200, 0.3250, 0.3300, 0.3400,
                                                0.3500, 0.3700, 0.4050, 0.4180, 0.4300, 0.4400, 0.4500, 0.4700, 0.5000, 0.5500,
                                                0.6500 };
        std::vector<ScalarFloat> attn_chi = {   0.1100, 0.1110, 0.1125, 0.1135, 0.1126, 0.1104, 0.1078, 0.1065, 0.1041, 0.0996,
                                                0.0971, 0.0939, 0.0896, 0.0859, 0.0823, 0.0788, 0.0746, 0.0726, 0.0690, 0.0660,
                                                0.0636, 0.0600, 0.0578, 0.0540, 0.0498, 0.0475, 0.0467, 0.0450, 0.0440, 0.0426,
                                                0.0410, 0.0400, 0.0390, 0.0375, 0.0360, 0.0340, 0.0330, 0.0328, 0.0325, 0.0330,
                                                0.0340, 0.0350, 0.0360, 0.0375, 0.0385, 0.0400, 0.0420, 0.0430, 0.0440, 0.0445,
                                                0.0450, 0.0460, 0.0475, 0.0490, 0.0515, 0.0520, 0.0505, 0.0440, 0.0390, 0.0340,
                                                0.0300 };
        std::vector<ScalarFloat> attn_e = {     0.668, 0.672, 0.680, 0.687, 0.693, 0.701, 0.707, 0.708, 0.707, 0.704,
                                                0.701, 0.699, 0.700, 0.703, 0.703, 0.703, 0.703, 0.704, 0.702, 0.700,
                                                0.700, 0.695, 0.690, 0.685, 0.680, 0.675, 0.670, 0.665, 0.660, 0.655,
                                                0.650, 0.645, 0.640, 0.630, 0.623, 0.615, 0.610, 0.614, 0.618, 0.622,
                                                0.626, 0.630, 0.634, 0.638, 0.642, 0.647, 0.653, 0.658, 0.663, 0.667,
                                                0.672, 0.677, 0.682, 0.687, 0.695, 0.697, 0.693, 0.665, 0.640, 0.620,
                                                0.600 };
        // IMPORTANT: This table uses the values provided by Morel, which are different than the ones from 6SV
        std::vector<ScalarFloat> molecular_scatter_coeffs = {   0.00618095, 0.00578095, 0.00547619, 0.00517619, 0.00492222, 
                                                                0.0046746 , 0.00447143, 0.00426825, 0.00406508, 0.0038619 , 
                                                                0.00365873, 0.00346667, 0.00331429, 0.0031619 , 0.00300952, 
                                                                0.00287143, 0.00276984, 0.00265238, 0.0025    , 0.00236508, 
                                                                0.00226349, 0.0021619 , 0.00206032, 0.00195873, 0.00185714, 
                                                                0.00177778, 0.00172698, 0.00167619, 0.0016254 , 0.0015746 , 
                                                                0.00152381, 0.00144603, 0.00134444, 0.0013    , 0.0013    , 
                                                                0.00126984, 0.00121905, 0.00116825, 0.00111746, 0.00107   , 
                                                                0.00102429, 0.00098556, 0.00095   , 0.0009181 , 0.00088762, 
                                                                0.00085714, 0.00082667, 0.00079619, 0.00076571, 0.00073937, 
                                                                0.00071397, 0.00069286, 0.00067254, 0.00065222, 0.0006319 , 
                                                                0.00061159, 0.00059127, 0.00057095, 0.00055063, 0.00053524, 
                                                                0.00052 };

        std::vector<ScalarFloat> molecular_scatter_coeffs_6s = {    0.0076, 0.0072, 0.0068, 0.0064, 0.0061,
                                                                    0.0058, 0.0055, 0.0052, 0.0049, 0.0047,
                                                                    0.0045, 0.0043, 0.0041, 0.0039, 0.0037,
                                                                    0.0036, 0.0034, 0.0033, 0.0031, 0.0030,
                                                                    0.0029, 0.0027, 0.0026, 0.0025, 0.0024,
                                                                    0.0023, 0.0022, 0.0022, 0.0021, 0.0020,
                                                                    0.0019, 0.0018, 0.0018, 0.0017, 0.0017,
                                                                    0.0016, 0.0016, 0.0015, 0.0015, 0.0014,
                                                                    0.0014, 0.0013, 0.0013, 0.0012, 0.0012,
                                                                    0.0011, 0.0011, 0.0010, 0.0010, 0.0010,
                                                                    0.0010, 0.0009, 0.0008, 0.0008, 0.0008,
                                                                    0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
                                                                    0.0007 };
 
        // Construct distributions from the provided data sets
        m_effective_reflectance = IrregularContinuousDistribution<ScalarFloat>(
            wc_wavelengths.data(), wc_data.data(), wc_data.size()
        );

        m_ior_real = IrregularContinuousDistribution<ScalarFloat>(
            ior_wavelengths.data(), ior_real_data.data(), ior_real_data.size()
        );

        m_ior_imag = IrregularContinuousDistribution<ScalarFloat>(
            ior_wavelengths.data(), ior_cplx_data.data(), ior_cplx_data.size()
        );

        m_attn_k = IrregularContinuousDistribution<ScalarFloat>(
            attn_wavelengths.data(), attn_k.data(), attn_k.size()
        );

        m_attn_chi = IrregularContinuousDistribution<ScalarFloat>(
            attn_wavelengths.data(), attn_chi.data(), attn_chi.size()
        );

        m_attn_e = IrregularContinuousDistribution<ScalarFloat>(
            attn_wavelengths.data(), attn_e.data(), attn_e.size()
        );

        m_molecular_scatter_coeffs = IrregularContinuousDistribution<ScalarFloat>(
            attn_wavelengths.data(), molecular_scatter_coeffs.data(), molecular_scatter_coeffs.size()
        );

        m_molecular_scatter_coeffs_6s = IrregularContinuousDistribution<ScalarFloat>(
            attn_wavelengths.data(), molecular_scatter_coeffs_6s.data(), molecular_scatter_coeffs_6s.size()
        );

        auto [azimuth_quad_pts, azimuth_quad_weights] = quad::gauss_legendre<FloatX>(AZIMUTH_QUADRATURE_PTS);
        auto [zenith_quad_pts, zenith_quad_weights] = quad::gauss_legendre<FloatX>(ZENITH_QUADRATURE_PTS);

        m_azimuth_pts = azimuth_quad_pts;
        m_azimuth_weights = azimuth_quad_weights;
        m_zenith_pts = zenith_quad_pts;
        m_zenith_weights = zenith_quad_weights;
    }

    /**
     * @brief Evaluate the effective reflectance of whitecaps.
     * 
     * Evaluates the effective reflectance of whitecaps at the given 
     * wavelength. The value returned already takes into account 
     * the base offset of 0.4 as described by (Koepke 1984).
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @return ScalarFloat The effective reflectance of whitecaps.
     */
    ScalarFloat effective_reflectance(const ScalarFloat &wavelength) const {
        return m_effective_reflectance.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the real index of refraction of water.
     * 
     * Evaluates the real index of refraction of water at the given
     * wavelength. The value returned is the real part of the complex
     * index of refraction as described by (Hale & Querry 1973).
     * 
     * @param wavelength The wavelength at which to evaluate the index of refraction.
     * @return ScalarFloat The real part of the index of refraction.
     */
    ScalarFloat ior_real(const ScalarFloat &wavelength) const {
        return m_ior_real.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the complex index of refraction of water.
     * 
     * Evaluates the complex index of refraction of water at the given
     * wavelength. The value returned is the imaginary part of the complex
     * index of refraction as described by (Hale & Querry 1973).
     * 
     * @param wavelength The wavelength at which to evaluate the index of refraction.
     * @return ScalarFloat The imaginary part of the index of refraction.
     */
    ScalarFloat ior_cplx(const ScalarFloat &wavelength) const {
        return m_ior_imag.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the K-term of the attenuation coefficient of water.
     * 
     * Evaluates the K-term of the attenuation coefficient of water at the given
     * wavelength. The value returned is the K-term as described by (Morel 1988).
     * 
     * @param wavelength The wavelength at which to evaluate the K-term of the attenuation coefficient.
     * @return ScalarFloat The K-term of the attenuation coefficient.
     */
    ScalarFloat attn_k(const ScalarFloat &wavelength) const {
        return m_attn_k.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the Chi-term of the attenuation coefficient of water.
     * 
     * Evaluates the Chi-term of the attenuation coefficient of water at the given
     * wavelength. The value returned is the Chi-term as described by (Morel 1988).
     * 
     * @param wavelength The wavelength at which to evaluate the Chi-term of the attenuation coefficient.
     * @return ScalarFloat The Chi-term of the attenuation coefficient.
     */
    ScalarFloat attn_chi(const ScalarFloat &wavelength) const {
        return m_attn_chi.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the E-term of the attenuation coefficient of water.
     * 
     * Evaluates the E-term of the attenuation coefficient of water at the given
     * wavelength. The value returned is the E-term as described by (Morel 1988).
     * 
     * @param wavelength The wavelength at which to evaluate the E-term of the attenuation coefficient.
     * @return ScalarFloat The E-term of the attenuation coefficient.
     */
    ScalarFloat attn_e(const ScalarFloat &wavelength) const {
        return m_attn_e.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the molecular scattering coefficient of water.
     * 
     * Evaluates the molecular scattering coefficient of water at the given
     * wavelength. The value returned is the molecular scattering coefficient
     * as described by (Morel 1988).
     * 
     * @param wavelength The wavelength at which to evaluate the molecular scattering coefficient.
     * @return ScalarFloat The molecular scattering coefficient.
     */
    ScalarFloat molecular_scatter_coeff(const ScalarFloat &wavelength) const {
        return m_molecular_scatter_coeffs.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the molecular scattering coefficient of water.
     * 
     * Evaluates the molecular scattering coefficient of water at the given
     * wavelength. The value returned is the molecular scattering coefficient
     * as described by (6S).
     * 
     * @param wavelength The wavelength at which to evaluate the molecular scattering coefficient.
     * @return ScalarFloat The molecular scattering coefficient.
     */
    ScalarFloat molecular_scatter_coeff_6s(const ScalarFloat &wavelength) const {
        return m_molecular_scatter_coeffs_6s.eval_pdf(wavelength);
    }

    /**
     * @brief Evaluate the azimuthal quadrature point and weight.
     * 
     * Evaluates the azimuthal quadrature point and weight at the given index.
     * The quadrature point has already been transformed to the range [0, 2π]. 
     * The quadrature weight has already been multiplied by the Jacobian of the
     * transformation.
     * 
     * @param i The index of the quadrature point.
     * @return std::tuple<Float, Float> The azimuthal quadrature point and weight.
     * @throw std::runtime_error If the index is out of bounds.
     * @note The azimuthal quadrature points are in the range [0, 2π].
     * @note The azimuthal quadrature weights are in the range [0, π].
     * @see AZIMUTH_QUADRATURE_PTS
     */
    std::tuple<Float, Float> eval_azimuth(ScalarUInt32 i) const {
        if (i > AZIMUTH_QUADRATURE_PTS)
            Throw("Invalid azimuth index");
        Float point = dr::Pi<Float> * (1.0f + m_azimuth_pts[i]);
        Float weight = dr::Pi<Float> * Float(m_azimuth_weights[i]);

        return { point, weight };
    }

    /**
     * @brief Evaluate the zenithal quadrature point and weight.
     * 
     * Evaluates the zenithal quadrature point and weight at the given index.
     * The quadrature point has already been transformed to the range [0, π/2].
     * The quadrature weight has already been multiplied by the Jacobian of the
     * transformation.
     * 
     * @param i The index of the quadrature point.
     * @return std::tuple<Float, Float> The zenithal quadrature point and weight.
     * @throw std::runtime_error If the index is out of bounds.
     * @note The zenithal quadrature points are in the range [0, π/2].
     * @note The zenithal quadrature weights are in the range [0, π/4].
     * @see ZENITH_QUADRATURE_PTS
     */
    std::tuple<Float, Float> eval_zenith(ScalarUInt32 i) const {
        if (i > ZENITH_QUADRATURE_PTS)
            Throw("Invalid zenith index");
        Float point = 0.25f * dr::Pi<Float> * (1.0f + m_zenith_pts[i]);
        Float weight = 0.25f * dr::Pi<Float> * Float(m_zenith_weights[i]);

        return { point, weight };
    }

private:
    // Effective reflectance of whitecaps
    IrregularContinuousDistribution<ScalarFloat> m_effective_reflectance;

    // Real/Complex IOR of water (Hale & Querry 1973)
    IrregularContinuousDistribution<ScalarFloat> m_ior_real;
    IrregularContinuousDistribution<ScalarFloat> m_ior_imag;

    // Water scattering and attenuation coefficients (Morel 1988)
    IrregularContinuousDistribution<ScalarFloat> m_attn_k;
    IrregularContinuousDistribution<ScalarFloat> m_attn_chi;
    IrregularContinuousDistribution<ScalarFloat> m_attn_e;
    IrregularContinuousDistribution<ScalarFloat> m_molecular_scatter_coeffs;
    IrregularContinuousDistribution<ScalarFloat> m_molecular_scatter_coeffs_6s;

    //  Better quadrature point storage
    FloatX m_azimuth_pts;
    FloatX m_azimuth_weights;
    FloatX m_zenith_pts;
    FloatX m_zenith_weights;
};

template<typename Float, typename Spectrum>
class OceanUtilities {
public:
    MI_IMPORT_TYPES()

    /**
     * @brief Construct a new Ocean Utilities object.
     * 
     * Construct a new Ocean Utilities object and initializes the ocean properties.
     */
    OceanUtilities() : m_ocean_props() { }

    /**
     * @brief Evaluate the fractional coverage of whitecaps.
     * 
     * Evaluates the fractional coverage of whitecaps at the given wind speed,
     * using the Monahan et al. (1986) model. The coverage is clamped to the
     * range [0, 1] (i.e. wind speed can be within the range [0, 37.54]).
     * 
     * @param wind_speed The wind speed at which to evaluate the coverage.
     * @return ScalarFloat The fractional coverage of whitecaps.
     */
    ScalarFloat eval_whitecap_coverage(const ScalarFloat &wind_speed) {
        return dr::clamp(m_monahan_alpha * dr::pow(wind_speed, m_monahan_lambda), 0.0f, 1.0f);
    }

    /**
     * @brief Evaluate the reflectance of whitecaps.
     * 
     * Evaluates the reflectance of whitecaps at the given wavelength and wind speed.
     * The reflectance is computed as the product of the effective reflectance of whitecaps
     * and the fractional coverage of whitecaps.
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @param wind_speed The wind speed at which to evaluate the reflectance.
     * @return ScalarFloat The reflectance of whitecaps.
     */
    ScalarFloat eval_whitecaps(const ScalarFloat &wavelength, const ScalarFloat &wind_speed) {
        // Compute the fractional coverage of whitecaps
        ScalarFloat coverage = eval_whitecap_coverage(wind_speed);

        // Proper interpolation of the effective reflectance
        ScalarFloat eff_reflectance = m_ocean_props.effective_reflectance(wavelength);

        // Compute the whitecap reflectance
        ScalarFloat whitecap_reflectance = coverage * eff_reflectance;

        return whitecap_reflectance;
    }

    /**
     * @brief Evaluate the sun glint reflectance.
     * 
     * Evaluates the sun glint reflectance at the given wavelength, incident and outgoing
     * directions, wind direction, wind speed, and chlorinity. The reflectance is computed
     * using the Cox-Munk distribution, the Fresnel equations, and relative tilt of the
     * oceanic surface.
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @param wind_direction The direction of the wind relative to the incident light.
     * @param wind_speed The wind speed at which to evaluate the reflectance.
     * @param chlorinity The chlorinity of the water.
     * @return ScalarFloat The sun glint reflectance.
     */
    Float eval_glint(const ScalarFloat &wavelength, 
                     const Vector3f &wi, const Vector3f &wo, 
                     const ScalarFloat &wind_direction, const ScalarFloat &wind_speed,
                     const ScalarFloat &chlorinity) {
        // Transform directions into azimuthal and zenithal angles
        Float theta_i = dr::acos(wi.z());
        Float theta_o = dr::acos(wo.z());
        Float phi_i = dr::atan2(wi.y(), wi.x());
        Float phi_o = dr::atan2(wo.y(), wo.x());
        Float phi = phi_i - phi_o;
        Float phi_w = phi_i - wind_direction;
        
        return eval_sun_glint(wavelength, theta_i, theta_o, phi, phi_w, wind_speed, chlorinity);
    }

    /**
     * @brief Evaluate the underwater light reflectance.
     * 
     * Evaluates the underwater light reflectance at the given wavelength, incident and outgoing
     * directions, wind direction, wind speed, chlorinity, and pigmentation. The reflectance is
     * computed by performing two quadratures to compute both upwelling and downwelling 
     * transmittance, attenuted with the ratio of upwell to downwelling irradiance.
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @param wind_direction The direction of the wind relative to the incident light.
     * @param wind_speed The wind speed at which to evaluate the reflectance.
     * @param chlorinity The chlorinity of the water.
     * @param pigmentation The pigmentation of the water.
     * @return ScalarFloat The underwater light reflectance.
     */
    Float eval_underlight(const ScalarFloat &wavelength, 
                          const Vector3f &wi, const Vector3f &wo,
                          const ScalarFloat &wind_direction, const ScalarFloat &wind_speed,
                          const ScalarFloat &chlorinity, const ScalarFloat &pigmentation) {
        // Transform directions into azimuthal and zenithal angles
        Float theta_i = dr::acos(wi.z());
        Float theta_o = dr::acos(wo.z());
        Float phi_i = dr::atan2(wi.y(), wi.x());
        Float phi_w = phi_i - wind_direction;

        return eval_underlight_angular(wavelength, theta_i, theta_o, phi_w, wind_speed, chlorinity, pigmentation);
    }

private:
    // Ocean properties
    OceanProperties<Float, Spectrum> m_ocean_props;

    // Whitecap constants
    const ScalarFloat m_f_eff_base = 0.4f;
    const ScalarFloat m_monahan_alpha = 2.95e-06f;
    const ScalarFloat m_monahan_lambda = 3.52f;

    // Cox-Munk distribution constants
    const ScalarFloat m_c_40 = 0.40f;
    const ScalarFloat m_c_22 = 0.12f;
    const ScalarFloat m_c_04 = 0.23f;

    // Underlight parameters
    const ScalarFloat m_underlight_alpha = 0.485f;

    /**
     * @brief Evaluate the Cox-Munk distribution.
     * 
     * Evaluates the Cox-Munk distribution at the given relative wind direction,
     * x and y components of the sensor/emitter direction, and wind speed. The distribution
     * is computed using the Cox-Munk model.
     * 
     * @param phi_w The relative wind direction.
     * @param z_x The x component of the sensor/emitter direction.
     * @param z_y The y component of the sensor/emitter direction.
     * @param wind_speed The wind speed at which to evaluate the distribution.
     * @return Float The probability of the Cox-Munk distribution.
     */
    Float eval_cox_munk(const Float &phi_w, 
                        const Float &z_x, const Float &z_y,
                        const ScalarFloat &wind_speed) const {
        Float sigma_c = 0.003f + 0.00192f * wind_speed;
        Float sigma_u = 0.00316f * wind_speed;

        Float c_21 = 0.01f - 0.0086f * wind_speed;
        Float c_03 = 0.04f - 0.033f * wind_speed;

        Float xe = (dr::cos(phi_w) * z_x + dr::sin(phi_w) * z_y) / dr::sqrt(sigma_c);
        Float xn = (-dr::sin(phi_w) * z_x + dr::cos(phi_w) * z_y) / dr::sqrt(sigma_u);

        Float xe2 = xe * xe;
        Float xn2 = xn * xn;
        
        Float coef = 1.0f - (c_21 / 2.0f) * (xe2 - 1.0f) * xn - (c_03 / 6.0f) * (xn2 - 3.0f) * xn;
        coef = coef + (m_c_40 / 24.0f) * (xe2 * xe2 - 6.0f * xe2 + 3.0f);
        coef = coef + (m_c_04 / 24.0f) * (xn2 * xn2 - 6.0f * xn2 + 3.0f);
        coef = coef + (m_c_22 / 4.0f) * (xe2 - 1.0f) * (xn2 - 1.0f);
        
        Float prob = coef / 2.0f / dr::Pi<Float> / dr::sqrt(sigma_u) / dr::sqrt(sigma_c) * dr::exp(-(xe2 + xn2) / 2.0f);
        return prob;
    }

    /**
     * @brief Evaluate the Fresnel coefficient.
     * 
     * Evaluates the Fresnel coefficient at the given real and imaginary parts of the
     * index of refraction, and relevant geometry terms derived from the incoming and
     * outgoing directions. The coefficient is computed using the Fresnel equations.
     * 
     * @param n_real The real part of the index of refraction.
     * @param n_imag The imaginary part of the index of refraction.
     * @param coschi The cosine of the geometry term.
     * @param sinchi The sine of the geometry term.
     * @return Float The Fresnel coefficient.
     */
    Float eval_fresnel(const ScalarFloat &n_real, const ScalarFloat &n_imag,
                       const Float &coschi, const Float &sinchi) const {
        Float s = (n_real * n_real) - (n_imag * n_imag) - (sinchi * sinchi);
        
        Float a_1 = dr::abs(s);
        Float a_2 = dr::sqrt(dr::sqr(s) + 4.0f * n_real * n_real * n_imag * n_imag);

        Float u = dr::sqrt(0.5f * dr::abs(a_1 + a_2));
        Float v = dr::sqrt(0.5f * dr::abs(a_2 - a_1));

        Float b_1 = (n_real * n_real - n_imag * n_imag) * coschi;
        Float b_2 = 2 * n_real * n_imag * coschi;

        Float right_squared = (dr::sqr(coschi - u) + v * v) / (dr::sqr(coschi + u) + v * v);
        Float left_squared = (dr::sqr(b_1 - u) + dr::sqr(b_2 + v)) / (dr::sqr(b_1 + u) + dr::sqr(b_2 - v));
        Float R = (right_squared + left_squared) * 0.5f;

        return R;
    }

    /**
     * @brief Evaluate the sun glint reflectance.
     * 
     * Evaluates the sun glint reflectance at the given wavelength, incident and outgoing
     * angles, relative azimuthal angle for ingoing and outgoing directions, relative azimuthal 
     * angle for wind direction and wind speed, and chlorinity. The reflectance is computed
     * using the Cox-Munk distribution, the Fresnel equations, and relative tilt of the
     * oceanic surface.
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @param theta_i The incident zenith angle.
     * @param theta_o The outgoing zenith angle.
     * @param phi The relative azimuthal angle.
     * @param phi_w The relative azimuthal angle for wind direction.
     * @param wind_speed The wind speed at which to evaluate the reflectance.
     * @param chlorinity The chlorinity of the water.
     * @param invert_real Whether to invert the real part of the IOR.
     * @return Float The sun glint reflectance.
     */
    Float eval_sun_glint(const ScalarFloat &wavelength, 
                         const Float &theta_i, const Float &theta_o,
                         const Float &phi,
                         const Float &phi_w, const ScalarFloat &wind_speed,
                         const ScalarFloat &chlorinity,
                         const bool invert_real = false) {
        // Implementation analog to 6SV
        Float c_i = dr::cos(theta_i);
        Float c_o = dr::cos(theta_o);
        Float s_i = dr::sin(theta_i);
        Float s_o = dr::sin(theta_o);

        Float z_x = (-s_o * dr::sin(phi)) / (c_i + c_o);
        Float z_y = (s_i + s_o * dr::cos(phi)) / (c_i + c_o);

        // Tilt angle (rad)
        Float tan_tilt = dr::sqrt(z_x * z_x + z_y * z_y);
        Float tilt = dr::atan(tan_tilt);

        // Cox-Munk specular probability
        Float specular_prob = eval_cox_munk(phi_w, z_x, z_y, wind_speed);
        auto mask = Mask(specular_prob < 0.0f);
        specular_prob = dr::select(mask, 0.0f, specular_prob);

        Float cos_2_chi = c_o * c_i + s_o * s_i * dr::cos(phi);
        auto ge_1 = Mask(cos_2_chi > 1.0f);
        auto le_1 = Mask(cos_2_chi < -1.0f);
        
        cos_2_chi = dr::select(ge_1, 0.999999999f, cos_2_chi);
        cos_2_chi = dr::select(le_1, -0.999999999f, cos_2_chi);
        
        Float coschi = dr::sqrt(0.5f * (1.0f + cos_2_chi));
        Float sinchi = dr::sqrt(0.5f * (1.0f - cos_2_chi));

        auto ge_coschi = Mask(coschi > 1.0f);
        auto le_coschi = Mask(coschi < -1.0f);
        auto ge_sinchi = Mask(sinchi > 1.0f);
        auto le_sinchi = Mask(sinchi < -1.0f);

        coschi = dr::select(ge_coschi, 0.999999999f, coschi);
        coschi = dr::select(le_coschi, -0.999999999f, coschi);
        sinchi = dr::select(ge_sinchi, 0.999999999f, sinchi);
        sinchi = dr::select(le_sinchi, -0.999999999f, sinchi);

        // Fresnel coefficient
        ScalarFloat n_real = m_ocean_props.ior_real(wavelength) + friedman_sverdrup(chlorinity);
        ScalarFloat n_imag = m_ocean_props.ior_cplx(wavelength);
        
        // Invert the real part of the IOR
        if (invert_real) {
            n_real = -n_real;
            n_imag = 0.0f;
        }

        Float fresnel_coeff = eval_fresnel(n_real, n_imag, coschi, sinchi);
        
        // Sun glint reflectance
        Float num = dr::Pi<Float> * fresnel_coeff * specular_prob;
        Float denom = 4.0f * c_i * c_o * dr::pow(dr::cos(tilt), 4.0f);

        return num / denom;
    }

    /**
     * @brief Evaluate the underlight reflectance.
     * 
     * Evaluates the underlight reflectance at the given wavelength, incident and outgoing
     * angles, relative azimuthal angle for wind direction, wind speed, chlorinity, and pigmentation.
     * The reflectance is computed by performing two quadratures to compute both upwelling and down-
     * welling transmittance, attenuted with the ratio of upwell to downwelling irradiance.
     * 
     * @param wavelength The wavelength at which to evaluate the reflectance.
     * @param theta_i The incident zenith angle.
     * @param theta_o The outgoing zenith angle.
     * @param phi_w The relative azimuthal angle for wind direction.
     * @param wind_speed The wind speed at which to evaluate the reflectance.
     * @param chlorinity The chlorinity of the water.
     * @param pigmentation The pigmentation of the water.
     * @return Value The underlight reflectance.
     */
    Float eval_underlight_angular(const ScalarFloat &wavelength, 
                          const Float &theta_i, const Float &theta_o,
                          const Float &phi_w, const ScalarFloat &wind_speed,
                          const ScalarFloat &chlorinity, const ScalarFloat &pigmentation) {
        // Analogue to 6SV, we return 0.0 if the wavelength is outside the range of [0.4, 0.7]
        auto outside_range = Mask(wavelength < 0.4f || wavelength > 0.7f);

        // Get IOR of water
        ScalarFloat n_real = m_ocean_props.ior_real(wavelength) + friedman_sverdrup(chlorinity);
        ScalarFloat n_imag = m_ocean_props.ior_cplx(wavelength);

        // Compute r_omega
        ScalarFloat r_om = r_omega(wavelength, pigmentation);

        // Upwelling and downwelling transmittance
        Float t_u = upwelling_transmittance(wavelength, n_real, theta_o, phi_w, wind_speed, chlorinity);
        Float t_d = downwelling_transmittance(wavelength, theta_i, phi_w, wind_speed, chlorinity);

        // Compute the underlight term
        Float underlight = (1.0f / (dr::sqr(n_real) + dr::sqr(n_imag))) * (r_om * t_u * t_d) / (1.0f - m_underlight_alpha * r_om);
        
        return dr::select(outside_range, 0.0f, underlight);
    }

    /**
     * @brief Compute the correction to the IOR of water.
     * 
     * Computes the correction to the index of refraction of water according to the
     * formulas provided by Friedman (1969) and Sverdrup (1942). The correction is
     * computed as a function of the chlorinity of the water.
     * 
     * @param chlorinity The chlorinity of the water.
     * @return ScalarFloat The correction to the index of refraction.
     */
    ScalarFloat friedman_sverdrup(const ScalarFloat &chlorinity) {
        return 0.00017492711f * (0.03f + 1.805f * chlorinity);
    }

    /**
     * @brief Compute the upwelling transmittance.
     * 
     * Computes the upwelling transmittance at the given wavelength, real part of the index of refraction,
     * outgoing zenith angle, relative azimuthal angle for wind direction, wind speed, and chlorinity. The
     * transmittance is computed by performing a quadrature over the hemisphere which evaluates the sun 
     * glint reflectance at each quadrature point.
     * 
     * @param wavelength The wavelength at which to evaluate the transmittance.
     * @param n_real The real part of the index of refraction.
     * @param theta_o The outgoing zenith angle.
     * @param phi_w The relative azimuthal angle for wind direction.
     * @param wind_speed The wind speed at which to evaluate the transmittance.
     * @param chlorinity The chlorinity of the water.
     * @return Float The upwelling transmittance.
     */
    Float upwelling_transmittance(const ScalarFloat &wavelength, 
                                  const ScalarFloat &n_real,
                                  const Float &theta_o,
                                  const Float &phi_w, const ScalarFloat &wind_speed,
                                  const ScalarFloat &chlorinity) {
        //  Transformation for the outgoing zenith angle, analogous to 6SV
        //  This accounts for Snell's law (?)
        Float t_w = dr::asin(dr::sin(theta_o) / n_real);

        //  Set up the quadrature accumulator
        Float tdv = 0.0f;
        Float summ = 0.0f;

        uint32_t loop_idx = 0;
        dr::scoped_set_flag guard(JitFlag::LoopRecord, false);
    
        dr::Loop<dr::mask_t<uint32_t>> quadrature_loop("Integrate over hemisphere", 
            loop_idx, tdv, summ);

        while(quadrature_loop(loop_idx < ZENITH_QUADRATURE_PTS * AZIMUTH_QUADRATURE_PTS)) {
                uint32_t azimuth_idx = loop_idx / ZENITH_QUADRATURE_PTS;
                uint32_t zenith_idx = loop_idx % ZENITH_QUADRATURE_PTS;

                auto [azimuth_point, azimuth_weight] = m_ocean_props.eval_azimuth(azimuth_idx);
                auto [zenith_point, zenith_weight] = m_ocean_props.eval_zenith(zenith_idx);

                Float geometry = dr::cos(zenith_point) * dr::sin(zenith_point);
                Float glint = eval_sun_glint(wavelength, t_w, zenith_point, azimuth_point, phi_w, wind_speed, chlorinity, true);
                Float geometryAndWeight = geometry * azimuth_weight * zenith_weight;

                //  Accumulate the result
                tdv += glint * geometryAndWeight;
                summ += geometryAndWeight;

                //  Increment the loop index
                loop_idx++;
            }

        return 1.0f - (tdv / summ);
    }

    /**
     * @brief Compute the downwelling transmittance.
     * 
     * Computes the downwelling transmittance at the given wavelength, incident zenith angle,
     * relative azimuthal angle for wind direction, wind speed, and chlorinity. The transmittance
     * is computed by performing a quadrature over the hemisphere which evaluates the sun glint
     * reflectance at each quadrature point.
     * 
     * @param wavelength The wavelength at which to evaluate the transmittance.
     * @param theta_i The incident zenith angle.
     * @param phi_w The relative azimuthal angle for wind direction.
     * @param wind_speed The wind speed at which to evaluate the transmittance.
     * @param chlorinity The chlorinity of the water.
     * @return Float The downwelling transmittance.
     */
    Float downwelling_transmittance(const ScalarFloat &wavelength, 
                                    const Float &theta_i,
                                    const Float &phi_w, const ScalarFloat &wind_speed,
                                    const ScalarFloat &chlorinity) {
        //  Set up the quadrature accumulator
        Float tds = 0.0f;
        Float summ = 0.0f;

        uint32_t loop_idx = 0;
        dr::scoped_set_flag guard(JitFlag::LoopRecord, false);
        dr::Loop<dr::mask_t<uint32_t>> quadrature_loop("Integrate over hemisphere", 
            loop_idx, tds, summ);

        while(quadrature_loop(loop_idx < ZENITH_QUADRATURE_PTS * AZIMUTH_QUADRATURE_PTS)) {
            uint32_t azimuth_idx = loop_idx / ZENITH_QUADRATURE_PTS;
            uint32_t zenith_idx = loop_idx % ZENITH_QUADRATURE_PTS;

            auto [azimuth_point, azimuth_weight] = m_ocean_props.eval_azimuth(azimuth_idx);
            auto [zenith_point, zenith_weight] = m_ocean_props.eval_zenith(zenith_idx);

            Float geometry = dr::cos(zenith_point) * dr::sin(zenith_point);
            Float glint = eval_sun_glint(wavelength, theta_i, zenith_point, azimuth_point, phi_w, wind_speed, chlorinity);
            Float geometryWeight = geometry * azimuth_weight * zenith_weight;

            //  Accumulate the result
            tds += glint * geometryWeight;
            summ += geometryWeight;

            //  Increment the loop index
            loop_idx++;
        }

        return 1.0f - (tds / summ);
    }

    /** 
     * @brief Compute the ratio of upwelling to downwelling irradiance.
     * 
     * Computes the ratio of upwelling to downwelling irradiance at the given wavelength
     * and pigmentation. The ratio is computed by performing an iterative computation.
     * 
     * @param wavelength The wavelength at which to evaluate the ratio.
     * @param pigmentation The pigmentation of the water.
     * @return ScalarFloat The ratio of upwelling to downwelling irradiance.
     * @note This function is only defined for wavelengths in the range [0.4, 0.7] nm.
     */
    ScalarFloat r_omega(const ScalarFloat &wavelength,
                        const ScalarFloat &pigmentation) {
        ScalarFloat wavelength_nm = wavelength * 1000.0f;
        ScalarFloat pigment_log = dr::log(pigmentation) / dr::log(10.0f);

        // Backscattering coefficient
        ScalarFloat molecular_scatter_coeff = m_ocean_props.molecular_scatter_coeff_6s(wavelength_nm);
        ScalarFloat scattering_coeff = 0.30f * dr::pow(pigmentation, 0.62);
        ScalarFloat backscatter_ratio = 0.002f + 0.02f * (0.5f - 0.25f * pigment_log) * (550.0 / wavelength_nm);
        ScalarFloat backscatter_coeff = 0.5f * molecular_scatter_coeff + scattering_coeff * backscatter_ratio;

        // (Diffuse) attenuation coefficient
        ScalarFloat k = m_ocean_props.attn_k(wavelength_nm);
        ScalarFloat chi = m_ocean_props.attn_chi(wavelength_nm);
        ScalarFloat e = m_ocean_props.attn_e(wavelength_nm);
        ScalarFloat attn_coeff = k + chi * dr::pow(pigmentation, e);

        // If any of the coefficients is zero, we return zero
        if (backscatter_coeff == 0.0f || attn_coeff == 0.0f)
            return 0.0f;

        // Iterative computation of the reflectance
        ScalarFloat u = 0.75f;
        ScalarFloat r_omega = 0.33f * backscatter_coeff / u / attn_coeff;

        bool converged = false;
        while (!converged) {
            // Update u
            u = (0.9f * (1.0f - r_omega)) / (1.0f + 2.25f * r_omega);

            // Update reflectance
            ScalarFloat r_omega_new = 0.33f * backscatter_coeff / (u * attn_coeff);

            // Create a mask that marks the converged values
            if (dr::abs((r_omega_new - r_omega) / r_omega_new) < 0.0001f) {
                converged = true;
                break;
            }

            // Update reflectance ONLY for non-converged values
            r_omega = r_omega_new;
        }

        return r_omega;
    }

};

#endif // OCEAN_PROPS

/**!

.. _plugin-bsdf-oceanic_legacy:

(Legacy 6S) Oceanic reflection model (:monosp:`oceanic-legacy`)
----------------------------------------------------

.. pluginparameters::

 * - component
   - |int|
   - Specifies which component of the oceanic reflection model to evaluate. Default: 0
        Component 0 is used to evaluate the total oceanic reflectance. Component 1 evaluates
        the whitecap reflectance. Component 2 evaluates the sun glint reflectance. Component 3
        evaluates the underlight reflectance. 

 * - wavelength
   - |float|
   - :math:`k \in [0.2, 4]`. 
   - Specifies the wavelength at which to evaluate the oceanic reflectance.

*  - wind_speed
   - |float|
   - :math:`k \in [0, 37.54]`. 
   - Specifies the wind speed at which to evaluate the oceanic reflectance.

*  - wind_direction
   - |float|
   - :math:`k \in [0, 2\pi]`.
   - Specifies the wind direction at which to evaluate the oceanic reflectance.

*  - chlorinity
   - |float|
   - Specifies the chlorinity of the water at which to evaluate the oceanic reflectance.

*  - pigmentation
   - |float|
   - :math:`k \in [0.3, \infty]`.
   - Specifies the pigmentation of the water at which to evaluate the oceanic reflectance.

*  - shininess
   - |float|
   - :math:`k \in [0, \infty]`.
   - Specifies the shininess which is used as the exponent for Blinn-Phong MIS.

This plugin implements the oceanic reflection model originally 
implemented in the 6S radiative transfer model. 

For the fundamental formulae defining the oceanic reflectance model, please refer to the
Eradiate Scientific Handbook.

Note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the ``twosided`` BSDF adapter plugin.
The following snippet describes an oceanic surface material with monochromatic parameters:

.. tab-set-code::

    .. code-block:: python

        "type": "oceanic-legacy",
        "channel": 0,
        "wavelength": 0.55,
        "wind_speed": 10,
        "wind_direction": 0,
        "chlorinity": 19,
        "pigmentation": 0.3,
        "shininess": 50

    .. code-block:: xml

        <bsdf type="ocenaic-legacy">
            <int name="component" value="0"/>
            <float name="wavelength" value="0.55"/>
            <float name="wind_speed" value="10"/>
            <float name="wind_direction" value="0"/>
            <float name="chlorinity" value="19"/>
            <float name="pigmentation" value="0.3"/>
            <float name="shininess" value="50"/>
        </bsdf>
*/
template <typename Float, typename Spectrum>
class OceanicBSDF final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    /**
     * @brief Construct a new OceanicBSDF object.
     * 
     * @param props A set of properties to initialize the oceanic BSDF.
     */
    OceanicBSDF(const Properties &props) : Base(props) {
        // Retrieve parameters
        m_component = props.get<ScalarInt32>("component");
        m_wavelength = props.get<ScalarFloat>("wavelength");
        m_wind_speed = props.get<ScalarFloat>("wind_speed");
        m_wind_direction = props.get<ScalarFloat>("wind_direction");
        m_chlorinity = props.get<ScalarFloat>("chlorinity");
        m_pigmentation = props.get<ScalarFloat>("pigmentation");
        m_shininess = props.get<ScalarFloat>("shininess");

        // Initialize the ocean utilities
        m_ocean_utils = new OceanUtilities<Float, Spectrum>();

        // Set the BSDF flags
        // => Whitecap and underlight reflectance is "diffuse"
        m_components.push_back(BSDFFlags::DiffuseReflection | 
                               BSDFFlags::FrontSide);
    
        // => Sun glint reflectance at the water surface is "specular"
        m_components.push_back(BSDFFlags::GlossyReflection | 
                               BSDFFlags::FrontSide);

        // Set all the flags
        for (auto c : m_components)
            m_flags |= c;
        dr::set_attr(this, "flags", m_flags);
    }

    /**
     * @brief Evaluate the whitecap reflectance.
     * 
     * Evaluates the whitecap reflectance at the provided wavelength and wind speed.
     * 
     * @return Float The whitecap reflectance.
     */
    Float eval_whitecaps() const {
        return m_ocean_utils->eval_whitecaps(m_wavelength, m_wind_speed);
    }
    
    /**
     * @brief Evaluate the sun glint reflectance.
     * 
     * Evaluates the sun glint reflectance at the provided wavelength, incident and outgoing
     * directions, wind direction, wind speed, and chlorinity.
     * 
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @return Float The sun glint reflectance.
     */
    Float eval_glint(const Vector3f &wi, const Vector3f &wo) const {
        return m_ocean_utils->eval_glint(m_wavelength, wi, wo, m_wind_direction, m_wind_speed, m_chlorinity);
    }

    /**
     * @brief Evaluate the underwater light reflectance.
     * 
     * Evaluates the underwater light reflectance at the provided wavelength, incident and outgoing
     * directions, wind direction, wind speed, chlorinity, and pigmentation.
     * 
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @return Float The underwater light reflectance.
     */
    Float eval_underlight(const Vector3f &wi, const Vector3f &wo) const {
        return m_ocean_utils->eval_underlight(m_wavelength, wi, wo, m_wind_direction, m_wind_speed, m_chlorinity, m_pigmentation);
    }

    /**
     * @brief Evaluate the oceanic reflectance.
     * 
     * Evaluates the oceanic reflectance at the provided wavelength, incident and outgoing
     * directions, wind direction, wind speed, chlorinity, and pigmentation. The reflectance is
     * computed by combining the whitecap, sun glint, and underwater light reflectance, according
     * to the coverage of whitecaps (based on 6S).
     * 
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @return Float The oceanic reflectance
     */
    Float eval_ocean(const Vector3f &wi, const Vector3f &wo) const {
        Float coverage = m_ocean_utils->eval_whitecap_coverage(m_wind_speed);
        Float whitecap_reflectance = eval_whitecaps();
        Float glint_reflectance = eval_glint(wi, wo);
        Float underlight_reflectance = eval_underlight(wi, wo);

        return (coverage * whitecap_reflectance) 
            + (1 - coverage) * glint_reflectance
            + (1 - (coverage * whitecap_reflectance)) * underlight_reflectance;
    }

    /**
     * @brief Evaluate the Blinn-Phong BRDF.
     * 
     * Evaluates the Blinn-Phong BRDF at the provided incident and outgoing directions, and
     * surface normal. The BRDF is computed using the Blinn-Phong distribution.
     * 
     * @param wi The incident direction of the light.
     * @param wo The outgoing direction of the light.
     * @param normal The surface normal.
     * @return Float The Blinn-Phong BRDF.
     * @note This function is not used, but serves as a reference for future implementations.
     */
    Float eval_blinn_phong(const Vector3f &wi, const Vector3f &wo, const Normal3f &normal) const {
        Float coverage = m_ocean_utils->eval_whitecap_coverage(m_wind_speed);
        Float factor = (m_shininess + 2.0f) / (2.0f * dr::Pi<Float>);
        Vector3f half = dr::normalize(wi + wo);
        Float dot = dr::dot(half, normal);

        // Blinn-phong => clamp dot to above zero
        dot = dr::clamp(dot, 0.0f, 1.0f);

        Float phong = dr::pow(dot, m_shininess);

        return coverage + (1 - coverage) * factor * phong;
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0);
        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        active &= cos_theta_i > 0.f;
        if (unlikely(dr::none_or<false>(active)) || (!has_diffuse && !has_specular))
            return { bs, 0.f };

        Float coverage = m_ocean_utils->eval_whitecap_coverage(m_wind_speed);
        Float prob_diff = coverage;

        Mask sample_diffuse = active & (sample1 < prob_diff),
             sample_glint = active && !sample_diffuse;

        if (dr::any_or<true>(sample_diffuse)) {
            // In the case of sampling the diffuse component, the outgoing direction
            // is sampled from a cosine-weighted hemisphere.
            dr::masked(bs.wo, sample_diffuse) = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.sampled_component, sample_diffuse) = 0;
            dr::masked(bs.sampled_type, sample_diffuse) = +BSDFFlags::DiffuseReflection;
        }

        if (dr::any_or<true>(sample_glint)) {
            // For Blinn-Phong, we need to sample the half-vector
            Float ksi_1 = sample2.x(),
                  ksi_2 = sample2.y();

            Float phi_h = dr::TwoPi<Float> * ksi_1;
            Float theta_h = dr::acos(dr::pow(ksi_2, 1.f / (m_shininess + 2.f)));

            Vector3f half = dr::normalize(Vector3f(dr::sin(theta_h) * dr::cos(phi_h),
                                                 dr::sin(theta_h) * dr::sin(phi_h),
                                                 dr::cos(theta_h)));

            Vector3f wo = 2.f * dr::dot(si.wi, half) * half - si.wi;

            // In the case of sampling the glint component, the outgoing direction
            // is sampled using the Blinn-Phong distribution.
            dr::masked(bs.wo, sample_glint) = wo; 
            dr::masked(bs.sampled_component, sample_glint) = 1;
            dr::masked(bs.sampled_type, sample_glint) = +BSDFFlags::GlossyReflection;
        }

        bs.pdf = pdf(ctx, si, bs.wo, active);
        bs.eta = 1.f;

        UnpolarizedSpectrum value = eval_ocean(si.wi, bs.wo) * Frame3f::cos_theta(bs.wo) / bs.pdf;

        return { bs, (depolarizer<Spectrum>(value)) & (active && bs.pdf > 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0);
        bool has_glint = ctx.is_enabled(BSDFFlags::GlossyReflection, 1);

        if (unlikely(dr::none_or<false>(active) ||  !has_glint && !has_diffuse))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Ensure incoming and outgoing directions are in the upper hemisphere
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        // Compute the whitecap reflectance
        UnpolarizedSpectrum result(0.f);
        UnpolarizedSpectrum blinn(0.f);
        UnpolarizedSpectrum whitecap_reflectance(0.f);
        UnpolarizedSpectrum glint_reflectance(0.f);
        UnpolarizedSpectrum underlight_reflectance(0.f);
        
        // Get the reflected directions
        auto is_reflect = Mask(dr::eq(dr::sign(cos_theta_i), dr::sign(cos_theta_o))) && active;

        if (has_diffuse) {
            // If whitecaps are enabled, compute the whitecap reflectance 
            whitecap_reflectance = eval_whitecaps();
            underlight_reflectance = eval_underlight(si.wi, wo);
        }
        
        if (has_glint)
            // If sun glint is enabled, compute the glint reflectance
            glint_reflectance = eval_glint(si.wi, wo);            
        
        // Combine the results
        Float coverage = m_ocean_utils->eval_whitecap_coverage(m_wind_speed);

        // For debugging purposes, the channel indicates what term of the BRDF to evaluate
        switch (m_component)
        {
            case 1:
                result[is_reflect] = (whitecap_reflectance) & active;
                break;
            case 2:
                result[is_reflect] = (1 - coverage) * glint_reflectance & active;
                break;
            case 3:
                result[is_reflect] = (1 - (coverage * whitecap_reflectance)) * underlight_reflectance & active;
                break;     
            default:
                result[is_reflect] = whitecap_reflectance
                    + (1 - coverage) * glint_reflectance
                    + (1 - (coverage * whitecap_reflectance)) * underlight_reflectance;
                
                // Cosine foreshortening factor
                result[is_reflect] *= cos_theta_o;
                break;
        }

        return depolarizer<Spectrum>(result) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 1);

        if (unlikely((!has_diffuse && !has_specular) || dr::none_or<false>(active)))
            return 0.f;

        Float coverage = m_ocean_utils->eval_whitecap_coverage(m_wind_speed);
        Float weight_diffuse = coverage,
              weight_specular = (1 - coverage);

        // Check if the normal has only zeros. If this is the case, use a default normal
        Vector3f normal = si.n;
        if (dr::all(normal == 0.f))
            normal = Vector3f(0.f, 0.f, 1.f);

        Vector3f half = dr::normalize(si.wi + wo);
        Float projection = dr::dot(half, normal);
        Float D = ((m_shininess + 2.0f) / dr::TwoPi<Float>) * dr::pow(projection, m_shininess);

        // We multiply the probability of the specular lobe with the pdf of 
        // the Blinn-Phong distribution and the probability of the diffuse lobe
        // with the pdf of the cosine-weighted hemisphere.
        Float pdf_diffuse = warp::square_to_cosine_hemisphere_pdf(wo),
              pdf_specular = (D * projection) / (4.0f * dr::dot(si.wi, half));

        Float pdf = weight_diffuse * pdf_diffuse + weight_specular * pdf_specular;

        // If the outgoing direction is in the lower hemisphere, we return zero
        Float cos_theta_o = Frame3f::cos_theta(wo);

        return dr::select(cos_theta_o > 0.f, pdf, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "OceanicLegacy[" << std::endl
            << "  component = " << string::indent(m_component) << std::endl
            << "  wavelength = " << string::indent(m_wavelength) << std::endl
            << "  wind_speed = " << string::indent(m_wind_speed) << std::endl
            << "  wind_direction = " << string::indent(m_wind_direction) << std::endl
            << "  chlorinity = " << string::indent(m_chlorinity) << std::endl
            << "  pigmentation = " << string::indent(m_pigmentation) << std::endl
            << "  shininess = " << string::indent(m_shininess) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    //  User-provided fields
    ScalarInt32 m_component;
    ScalarFloat m_wavelength;
    ScalarFloat m_wind_speed;
    ScalarFloat m_wind_direction;
    ScalarFloat m_chlorinity;
    ScalarFloat m_pigmentation;
    ScalarFloat m_shininess;

    //  Pointer to the ocean utilities
    OceanUtilities<Float, Spectrum> *m_ocean_utils;
};

MI_IMPLEMENT_CLASS_VARIANT(OceanicBSDF, BSDF)
MI_EXPORT_PLUGIN(OceanicBSDF, "Oceanic material")
NAMESPACE_END(mitsuba)