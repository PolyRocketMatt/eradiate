import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mi.set_variant("llvm_ad_rgb_double")

WHITECAP=0
GLINT=1
UNDERLIGHT=2
TOTAL=3

channels = [WHITECAP, GLINT, UNDERLIGHT, TOTAL]

res = 25
azim_vs = np.linspace(0, 2 * np.pi, 2 * res)
zen_vs = np.linspace(0, np.deg2rad(89), res)

wavelength = 2.2
incoming = 15
solar_azimuth = 0
wind_speed = 10
wind_direction = np.deg2rad(0)
n_real = 1.333
n_imag = 0.0
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
        'visual_type': 0,
        'wavelength': wavelength,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'chlorinity': chlorinity,
        'pigmentation': pigmentation,
        'shininess': 50,
    })
    
    #result = bsdf.eval(mi.BSDFContext(), si, wo)
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
zeros = np.count_nonzero(brdf_data == 0)

print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Mean excluding zeros: {mean_exluding_zeros}")
print(f"Zeros: {zeros}, Total: {res * 2 * res}, Percentage: {zeros / (res * 2 * res) * 100}%")

# Plot values for spherical coordinates
fig, ax = plt.subplots(figsize=(8, 4))
visual_set = [foam_data_vis, glint_data_vis, underlight_data_vis, brdf_data]
channel = 3
im = ax.imshow(visual_set[channel], interpolation='spline36', extent=[0, 2 * np.pi, np.pi / 2, 0], cmap='turbo')

# Name the axes
plt.xlabel("Outgoing Azimuth")
plt.ylabel("Outgoing Zenith")

# Add the title
degrees = np.degrees(incoming)

# Round to 2 decimal places
degrees = round(degrees, 2)

# Create x_axis tick labels in degrees
x_ticks = np.linspace(0, 2 * np.pi, 5)
x_labels = np.degrees(x_ticks)
x_labels = [round(label, 2) for label in x_labels]
plt.xticks(x_ticks, x_labels)

# Create y_axis tick labels in degrees
y_ticks = np.linspace(0, np.pi / 2, 3)
y_labels = np.degrees(y_ticks)
y_labels = [round(label, 2) for label in y_labels]
plt.yticks(y_ticks, y_labels)

# Add Title
ax.set_title(f'BRDF | Incoming Zenith: {incoming}°')

# Add color bar with same height as the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.25)
plt.colorbar(im, cax=cax)
plt.show()