import drjit as dr
import mitsuba as mi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb_double")

WHITECAP=0
GLINT=1
UNDERLIGHT=2
TOTAL=3

channels = [WHITECAP, GLINT, UNDERLIGHT, TOTAL]
wind_speeds = np.arange(0.1, 37.2455, 2.0)

def run(parametrization):
    results = {}
    progress = tqdm(total=len(wind_speeds), desc='Parametrization Progress')
    idx = 0

    for wind_speed in wind_speeds:
        key = f'eval_{idx}'
        res = 25

        wavelength = 2.2
        incoming = 15
        solar_azimuth = 0
        wind_direction = np.deg2rad(0)
        chlorinity = 19
        pigmentation = 0.3

        def sph_to_eucl(theta, phi):
            st, ct = dr.sincos(theta)
            sp, cp = dr.sincos(phi)
            return mi.Vector3f(cp * st, sp * st, ct)

        brdf_values = None
        brdf_foam = None
        brdf_glint = None
        brdf_underlight = None

        # Create a dummy surface interaction to use for the evaluation of the BSDF
        si = dr.zeros(mi.SurfaceInteraction3f)

        # Specify an incident direction with X degrees
        si.wi = sph_to_eucl(dr.deg2rad(incoming), dr.deg2rad(solar_azimuth))

        # Set the surface normal to point upwards
        si.n = mi.Vector3f(0, 0, 1)

        # Create grid in spherical coordinates and map it into a sphere
        res = 500
        zeniths_o, azimuths_o = dr.meshgrid(
            dr.linspace(mi.Float, np.deg2rad(1), np.deg2rad(89),     res),
            dr.linspace(mi.Float, 0, 2 * dr.pi, 2 * res)
        )
        wo = sph_to_eucl(zeniths_o, azimuths_o)

        for channel in channels:
            bsdf = mi.load_dict({
                'type': 'oceanic_legacy',
                'channel': channel,
                'visual_type': 3,
                'wavelength': wavelength,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'chlorinity': chlorinity,
                'pigmentation': pigmentation,
                'shininess': parametrization(wind_speed),
            })
            
            if channel == WHITECAP:
                brdf_foam = bsdf.eval(mi.BSDFContext(), si, wo)
            elif channel == GLINT:
                brdf_glint = bsdf.eval(mi.BSDFContext(), si, wo)
            elif channel == UNDERLIGHT:
                brdf_underlight = bsdf.eval(mi.BSDFContext(), si, wo)
            elif channel == TOTAL:
                brdf_values = bsdf.eval(mi.BSDFContext(), si, wo)

        brdf_np = np.array(brdf_values)
        foam_np = np.array(brdf_foam)
        glint_np = np.array(brdf_glint)
        underlight_np = np.array(brdf_underlight)

        brdf_data = brdf_np[:,0]
        foam_data = foam_np[:,0]
        glint_data = glint_np[:,0]
        underlight_data = underlight_np[:,0]

        # Extract red channel of BRDF values and reshape into 2D grid
        brdf_data = brdf_data.reshape(2 * res, res).T
        foam_data_vis = foam_data.reshape(2 * res, res).T
        glint_data_vis = glint_data.reshape(2 * res, res).T
        underlight_data_vis = underlight_data.reshape(2 * res, res).T

        dimensional_data = brdf_data.T
        foam_buffer = foam_data_vis.T
        glint_buffer = glint_data_vis.T
        underlight_buffer = underlight_data_vis.T

        width = len(dimensional_data)
        height = len(dimensional_data[0])

        foam_data = np.zeros((width, height))
        glint_data = np.zeros((width, height))
        underlight_data = np.zeros((width, height))

        for x in range(width):
            for y in range(height):
                f = foam_buffer[x][y]
                g = glint_buffer[x][y]
                u = underlight_buffer[x][y]
                total_val = dimensional_data[x][y]
                foam_comp = f / total_val
                glint_comp = g / total_val
                underlight_comp = u / total_val

                foam_data[x][y] = foam_comp
                glint_data[x][y] = glint_comp
                underlight_data[x][y] = underlight_comp

        # Min/Max values
        min_val = np.min(brdf_data)
        max_val = np.max(brdf_data)
        mean_val = np.mean(brdf_data)
        mean_exluding_zeros = np.mean(brdf_data[brdf_data != 0])

        results[key] = {
            'wind_speed': wind_speed,
            'min_val': min_val,
            'max_val': max_val,
            'mean_val': mean_val,
            'mean_exluding_zeros': mean_exluding_zeros
        }

        idx += 1
        progress.update(1)

    progress.close()

    # Compute averages
    avg_min = np.mean([results[str(index)]['min_val'] for index in results])
    avg_max = np.mean([results[str(index)]['max_val'] for index in results])
    avg_mean = np.mean([results[str(index)]['mean_val'] for index in results])
    avg_mean_exluding_zeros = np.mean([results[str(index)]['mean_exluding_zeros'] for index in results])

    print(f'Parametrization Report:')
    print(f'    Minimum Value: {avg_min}')
    print(f'    Maximum Value: {avg_max}')
    print(f'    Mean Value: {avg_mean}')
    print(f'    Mean Value (Excluding Zeros): {avg_mean_exluding_zeros}')

    # Make a plot for the mean error per wind speed
    ws = [results[str(index)]['wind_speed'] for index in results]
    mean_vals = [results[str(index)]['mean_val'] for index in results]

    plt.plot(ws, mean_vals)
    plt.xlabel('Wind Speed')
    plt.ylabel('Mean Value')
    plt.title('Mean Value per Wind Speed')
    plt.show()

def parametrization(wind_speed):
    return (37.2455 - wind_speed) ** 1.1

run(parametrization)