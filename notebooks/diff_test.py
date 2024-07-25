import numpy as np
import json
import os
import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable

mi.set_variant("llvm_ad_rgb_double")

def relative_diff(a, b):
    return (a - b) / b

def simple_diff(a, b):
    return a - b

def abs_diff(a, b):
    return np.abs(a - b)

def sph_to_eucl(theta, phi):
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

wind_speeds = [1, 10, 37]
wind_azimuth = 0
comparison_data = {}

incoming_zeniths = np.linspace(0, np.deg2rad(89), 18)
incoming_limit = np.deg2rad(30)

for wind_speed in wind_speeds:
    data_file = f'data/data_{wind_speed}ms.json'
    data = None

    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # BSDF Construction where we keep the same parameters as the generated data
    bsdf = mi.load_dict({
        'type': 'oceanic_legacy',
        'wavelength': 0.5,
        'wind_speed': wind_speed,
        'wind_direction': wind_azimuth,
        'chlorinity': 19,
        'pigmentation': 0.3
    })

    differences = []

    keys_as_floats = list(map(float, data.keys()))

    # Comparison loop for each solar zenith 
    for solar_zenith_key in data.keys():
        solar_zenith = float(solar_zenith_key)
        if solar_zenith > incoming_limit:
            continue

        # Get the reflectance data for the current solar zenith
        current_data = data[solar_zenith_key]
        zeniths = list(map(float, current_data.keys()))
        zenith_keys = current_data.keys()
        reflectances = []

        for key in zenith_keys:
            reflectance = current_data[key]
            reflectances.append(reflectance['total'])

        # Create a mitsuba scene to create similar data
        # Create a surface interaction
        si = dr.zeros(mi.SurfaceInteraction3f)

        # Create the solar direction. Data was generated at a solar zenith of 0 radians
        si.wi = sph_to_eucl(solar_zenith, dr.deg2rad(0))

        # Since we want to compare, we create a simple meshgrid of len(outgoing_zeniths) x 1
        resolution = len(zeniths)
        zeniths_o, azimuths_o = dr.meshgrid(
            dr.linspace(mi.Float, np.deg2rad(0), np.deg2rad(89), resolution),

            # Data was generated with a viewing azimuth of π radians
            dr.linspace(mi.Float, np.pi, np.pi, 1)
        )

        # Construct the outgoing directions
        wo = sph_to_eucl(zeniths_o, azimuths_o)

        # Evaluate the BSDF
        bsdf_values = bsdf.eval(mi.BSDFContext(), si, wo)
        bsdf_array = np.array(bsdf_values)

        # Extract one channel
        bsdf_array = bsdf_array[:, 0]

        # Compute the difference between the data and the Mitsuba BSDF
        reflectances = np.array(reflectances)
        reflectance_diff = abs_diff(reflectances, bsdf_array)
        total_diff = np.sum(reflectance_diff)
        avg_diff = np.mean(reflectance_diff)
        differences.append(avg_diff) 

        # Create a comparison plot between the data and the Mitsuba BSDF
        

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(zeniths, reflectances, label='Data')
        ax.plot(zeniths, bsdf_array, label='Mitsuba BSDF')
        ax.set_xlabel('Outgoing Zenith')
        ax.set_ylabel('BRDF Value')
        ax.set_title(f'Solar Zenith: {round(np.rad2deg(solar_zenith), 2)}')
        
        # Save the plot
        plt.legend()

        # Save the plot
        plt.savefig(f'plots/comp_{wind_speed}_{solar_zenith}.png')
        plt.close()

        #print(f'For solar zenith {solar_zenith}:')
        #print(f"  Total difference for solar zenith {solar_zenith}: {total_diff}")
        #print(f"  Average difference for solar zenith {solar_zenith}: {avg_diff}")
        #print(f'\n')
    comparison_data[wind_speed] = differences

    limited_solar_zeniths = [zenith for zenith in incoming_zeniths if zenith <= incoming_limit]

    filesnames = [f'plots/comp_{wind_speed}_{zenith}.png' for zenith in limited_solar_zeniths]

    images = [imageio.imread(filename) for filename in filesnames]
    imageio.mimsave(f'plots/comp_{wind_speed}.gif', images, duration=5)

# Incoming zeniths to degrees
incoming_zeniths = np.rad2deg(incoming_zeniths)

# Filter those that are above the incoming limit
incoming_zeniths = incoming_zeniths[incoming_zeniths <= np.rad2deg(incoming_limit)]

# Plot the differences for each wind speed relative to the incoming zeniths
fig, ax = plt.subplots()

for wind_speed in wind_speeds:
    ax.plot(incoming_zeniths, comparison_data[wind_speed], label=f'{wind_speed}m/s')

ax.set_xlabel('Solar Zenith')
ax.set_ylabel('Relative Difference')
ax.set_title('Relative Difference in Reflectance')
ax.legend()

plt.show()