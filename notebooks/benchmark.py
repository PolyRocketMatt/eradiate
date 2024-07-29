import json
import os
import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from drjit import *
from tqdm import tqdm

# Load Mitsuba
mi.set_variant('llvm_ad_rgb_double')

FOAM=0
GLINT=1
UNDERLIGHT=3
REFLECTANCE=4


SELECTED_CHANNEL = 4

wind_speeds = [0.1, 1, 10, 37]
wavelengths = [0.4, 0.5, 0.7, 1.5, 2.2]

solar_azimuth = 0
view_azimuth = 0
wind_direction = 0
salinity = 0
chlorinity = 0
pigmentation = 0

# Palettes
red_palette = ['#E06666'] * 45
green_palette = ['#9DBB53'] * 45
blue_palette = ['#78C7FF'] * 45
yellow_palette = ['#FFCA46'] * 45
palettes = [red_palette, yellow_palette, green_palette, blue_palette]

# From https://gist.github.com/notbanker/2be3ed34539c86e22ffdd88fd95ad8bc
class ChainedAssignent:
    """ Context manager to temporarily set pandas chained assignment warning. Usage:
    
        with ChainedAssignment():
             blah  
             
        with ChainedAssignment('error'):
             run my code and figure out which line causes the error! 
    
    """

    def __init__(self, chained = None):
        acceptable = [ None, 'warn','raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__( self ):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

# Helper to go from spherical to Euclidean coordinates
def sph_to_eucl(theta, phi):
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

# Helper function to compute the relative error
def relative_error(x, y):
    if y == 0:
        return x
    return (x - y) / y

def absolute_errpr(x, y):
    return np.abs(x - y)

# Channel Conversion 
def convert_channel(channel):
    if channel == FOAM:
        return 'foam'
    elif channel == GLINT:
        return 'glint'
    elif channel == UNDERLIGHT:
        return 'underlight'
    elif channel == REFLECTANCE:
        return 'reflectance'
    else:
        return 'unknown'

def load_data(wavelength):
    df = pd.DataFrame(columns=['solar_zenith', 'view_zenith', 'wind_speed', 'foam', 'glint', 'underlight', 'reflectance', 'mitsuba_channel', 'relative_error', 'absolute_error'])
    benchmark_data = {}
    settings = {}
    for wind_speed in wind_speeds:
        with open(f'data/data_{wind_speed}ms_{wavelength}.json') as file:
            data = json.load(file)

            # Obtain the other parameters from the data
            solar_azimuth = data['solar_azimuth']
            view_azimuth = data['view_azimuth']
            wind_direction = data['wind_direction']
            salinity = data['salinity']
            chlorinity = data['chlorinity']
            pigmentation = data['pigmentation']

            settings[wind_speed] = {
                'solar_azimuth': solar_azimuth,
                'view_azimuth': view_azimuth,
                'wind_direction': wind_direction,
                'salinity': salinity,
                'chlorinity': chlorinity,
                'pigmentation': pigmentation
            }

            # Solar Data
            solar_data = data['data']

            # Construct the data points that have to be plotted
            benchmark_data[wind_speed] = {}
            for solar_key, solar_key_data in solar_data.items():
                view_benchmarks = []
                for _, view_key_data in solar_key_data.items():
                    view_zenith = view_key_data['outgoing_zenith']
                    foam = view_key_data['foam']
                    glint = view_key_data['glint']
                    underlight = view_key_data['water']
                    reflectance = view_key_data['total']

                    # Set the dataframe
                    df = pd.concat([df, pd.DataFrame({
                        'solar_zenith': [float(solar_key)], 
                        'view_zenith': [view_zenith], 
                        'wind_speed': [wind_speed], 
                        'foam': [foam], 
                        'glint': [glint], 
                        'underlight': [underlight], 
                        'reflectance': [reflectance],
                        'mitsuba_channel': [np.nan],
                        'relative_error': [np.nan],
                        'absolute_error': [np.nan]
                    })], ignore_index=True)

                    # Benchmark Data
                    view_benchmarks.append(view_zenith)
                benchmark_data[wind_speed][solar_key] = view_benchmarks
    return df, benchmark_data, settings 


def benchmark(df, benchmark_data, settings, wavelength):
    for wind_speed in wind_speeds:
        # Collect the current settings
        ws_settings = settings[wind_speed]
        solar_azimuth = ws_settings['solar_azimuth']
        view_azimuth = ws_settings['view_azimuth']
        wind_direction = ws_settings['wind_direction']
        salinity = ws_settings['salinity']
        chlorinity = ws_settings['chlorinity']
        pigmentation = ws_settings['pigmentation']

        # Benchmarking
        ws_benchmark_data = benchmark_data[wind_speed]
        progress = tqdm(ws_benchmark_data.items(), desc=f'Wind Speed: {wind_speed} m/s')
        for solar_zenith, viewing_zeniths in ws_benchmark_data.items():
            solar_zenith = float(solar_zenith)

            # Create the surface interaction
            si = dr.zeros(mi.SurfaceInteraction3f)
            
            # Create the incident direction
            si.wi = sph_to_eucl(solar_zenith, solar_azimuth)

            min_zenith = np.min(viewing_zeniths)
            max_zenith = np.max(viewing_zeniths)

            zenith_array = dr.linspace(mi.Float, min_zenith, max_zenith, len(viewing_zeniths))
            azimuth_array = dr.full(mi.Float, view_azimuth)
        
            # Create data points
            bm_zeniths, bm_azimuths = dr.meshgrid(
                zenith_array,
                azimuth_array
            )
            wo = sph_to_eucl(bm_zeniths, bm_azimuths)

            # Construct BxDF
            bsdf = mi.load_dict({
                'type': 'oceanic_legacy',
                'channel': SELECTED_CHANNEL,
                'wavelength': wavelength,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'chlorinity': chlorinity,
                'pigmentation': pigmentation
            })

            # Evaluate the data points for the selected channel
            benchmark_eval = bsdf.eval(mi.BSDFContext(), si, wo)

            # Conversion to readable format and NumPy
            benchmark_eval = (np.array(benchmark_eval))[:,0]

            # Update the dataframe with the Mitsuba data
            for i, benchmark_eval_data in enumerate(benchmark_eval):
                df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), 'mitsuba_channel'] = benchmark_eval_data

                # Compute the relative error in the reference and mitsuba channels
                channel_name = convert_channel(SELECTED_CHANNEL)

                ref_channel = df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), channel_name].values[0]
                mitsuba_channel = df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), 'mitsuba_channel'].values[0]

                # Compute the relative and absolute error
                relative_err = relative_error(mitsuba_channel, ref_channel)
                absolute_err = absolute_errpr(mitsuba_channel, ref_channel)

                # Update the dataframe with the relative and absolute error
                df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), 'relative_error'] = relative_err
                df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), 'absolute_error'] = absolute_err

            # Update progress bar
            progress.update(1)
        progress.close()


def process_angles(df):
    # Add a column 'theta_i' in Latex format for the solar zenith in degrees
    df['t_i'] = df['solar_zenith'].apply(lambda x: f'{np.rad2deg(x):.2f}')
    df['t_o'] = df['view_zenith'].apply(lambda x: f'{np.rad2deg(x):.0f}')


def create_facet_plot(wavelength, wind_speed, palette, df, key='relative_error'):
    with ChainedAssignent():
        # Get the data for the current wind speed
        df = df[df['wind_speed'] == wind_speed]

        # Add a column which is a copy of the keyed error
        df['normalized_error'] = df[key]

        for solar_zenith in df['solar_zenith'].unique():
            mask = df['solar_zenith'] == solar_zenith
            normalization_constant = df.loc[mask]['reflectance'].max()
            df.loc[mask, 'normalized_error'] /= normalization_constant

        # Initialize the grid
        grid = sns.FacetGrid(df, col='t_i', hue='t_i', palette=palette,
                            col_wrap=5, height=2.0)
        
        # Draw the reference line
        grid.refline(y=0, linestyle=':')

        # Draw a line plot to show the trajectory of each relative error, but space 
        # the relative error such that it is defined over the view zenith
        grid.map(sns.lineplot, 't_o', 'normalized_error', errorbar=None)

        # Draw the vertical line for solar zenith
        def add_vertical_line(data, **kwargs):
            solar_zenith = (np.rad2deg(data['solar_zenith'].iloc[0]) / 90) * 18
            plt.axvline(x=solar_zenith, color='gray', linestyle='--')

        grid.map_dataframe(add_vertical_line)

        # Adjust the tick positions and labels
        grid.set(xticks=np.arange(0, 19, 6),
                xlim=(0, 18))

        # Adjust the arrangement of the plots
        grid.fig.tight_layout(w_pad=1)

        # Make directory for plots/error/wavelength
        if not os.path.exists(f'plots/error/{str(wavelength)}'):
            os.makedirs(f'plots/error/{str(wavelength)}')

        # Save the plot
        grid.savefig(f'plots/error/{str(wavelength)}/error_{wind_speed}ms_{wavelength}.pdf')


def run_complete_benchmark(wavelength):
    # Load the data
    df, benchmark_data, settings = load_data(wavelength)

    # Perform the benchmark
    benchmark(df, benchmark_data, settings, wavelength)

    # Process solar/viewing angles
    process_angles(df)

    # Create the facet plot for each wind speed
    progress = tqdm(wind_speeds, desc='Creating Facet Plots')
    for wind_speed in wind_speeds:
        palette = palettes[wind_speeds.index(wind_speed)]
        create_facet_plot(wavelength, wind_speed, palette, df, key='absolute_error')
        progress.update(1)


if not os.path.exists(f'plots/error'):
    os.makedirs(f'plots/error')

# Run the benchmark for each wavelength
total_progress = tqdm(wavelengths, desc='Running Benchmark')
for wavelength in wavelengths:
    print(f'Running benchmark for wavelength: {wavelength}')
    run_complete_benchmark(wavelength)
    total_progress.update(1)

