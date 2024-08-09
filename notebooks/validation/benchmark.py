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

# Possible components to compare
REFLECTANCE = 0
FOAM=1
GLINT=2
UNDERLIGHT=3
COMPONENTS = [REFLECTANCE, FOAM, GLINT, UNDERLIGHT]

# Comparison parameters
wind_speeds = [0.1, 1, 10, 37]
wavelengths = [0.5] #[0.4, 0.5, 0.7, 1.5, 2.2]
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
    '''
    Helper function to convert spherical coordinates to Euclidean coordinates
    '''
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

# Helper function to compute the relative error
def relative_error(x, y):
    '''
    Helper function to compute the relative error
    '''
    if y == 0:
        return x
    return (x - y) / y

def absolute_error(x, y):
    '''
    Helper function to compute the absolute error
    '''
    return np.abs(x - y)

# Channel Conversion 
def convert_channel(channel):
    '''
    Helper function to convert a channel (i.e. component) to a keyable string
    '''
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
    '''
    Load the data from the JSON files and construct the data points
    '''
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
                'pigmentation': pigmentation,
                'shininess': 20
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


def benchmark(df, benchmark_data, settings, wavelength, components):
    '''
    Perform the benchmark for the given data, settings and wavelength
    '''
    for wind_speed in wind_speeds:
        # Collect the current settings
        ws_settings = settings[wind_speed]
        solar_azimuth = ws_settings['solar_azimuth']
        view_azimuth = ws_settings['view_azimuth']
        wind_direction = ws_settings['wind_direction']
        salinity = ws_settings['salinity']
        chlorinity = ws_settings['chlorinity']
        pigmentation = ws_settings['pigmentation']
        shininess = ws_settings['shininess']

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

            for component in COMPONENTS:
                # Construct BxDF
                bsdf = mi.load_dict({
                    'type': 'oceanic_legacy',
                    'component': component,
                    'wavelength': wavelength,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    'chlorinity': chlorinity,
                    'pigmentation': pigmentation,
                    'shininess': shininess
                })

                # Evaluate the data points for the selected channel
                benchmark_eval = bsdf.eval(mi.BSDFContext(), si, wo)

                # Conversion to readable format and NumPy
                benchmark_eval = (np.array(benchmark_eval))[:,0]

                # Update the dataframe with the Mitsuba data
                for i, benchmark_eval_data in enumerate(benchmark_eval):
                    # Compute the relative error in the reference and mitsuba channels
                    component_name = convert_channel(component)

                    # Set the Mitsuba channel
                    df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), f'mitsuba_{component_name}'] = benchmark_eval_data

                    ref_channel = df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), component_name].values[0]
                    mitsuba_channel = df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), f'mitsuba_{component_name}'].values[0]

                    # Compute the relative and absolute error
                    relative_err = relative_error(mitsuba_channel, ref_channel)
                    absolute_err = absolute_error(mitsuba_channel, ref_channel)

                    # Update the dataframe with the relative and absolute error
                    df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), f'{component_name}_rel_error'] = relative_err
                    df.loc[(df['solar_zenith'] == solar_zenith) & (df['view_zenith'] == viewing_zeniths[i]) & (df['wind_speed'] == wind_speed), f'{component_name}_abs_error'] = absolute_err

            # Update progress bar
            progress.update(1)
        progress.close()


def process_angles(df):
    '''
    Conversion from radians to degrees and add the columns to the dataframe
    '''
    # Add a column 'theta_i' in Latex format for the solar zenith in degrees
    df['theta_i'] = df['solar_zenith'].apply(lambda x: f'{np.rad2deg(x):.2f}')
    df['theta_o'] = df['view_zenith'].apply(lambda x: f'{np.rad2deg(x):.0f}')


def process_errors(df, components):
    '''
    Add the relative and absolute error in percentage to the dataframe
    '''
    for component in components:
        component_name = convert_channel(component)

        # Add a column for the relative and absolute error in percentage
        df[f'{component_name}_rel_error_percentage'] = df[f'{component_name}_rel_error'] * 100
        df[f'{component_name}_abs_error_percentage'] = df[f'{component_name}_abs_error'] * 100


def process_contributions(df):
    '''
    Add percentage contributions
    '''
    sum = df['mitsuba_foam'] + df['mitsuba_glint'] + df['mitsuba_underlight']
    df['mitsuba_foam_contrib'] = df['mitsuba_foam'] / sum
    df['mitsuba_glint_contrib'] = df['mitsuba_glint'] / sum
    df['mitsuba_underlight_contrib'] = df['mitsuba_underlight'] / sum


def create_facet_plot(name, wavelength, wind_speed, df, key='reflectance_relative_error', tag='Relative Reflectance Error', ymin=-0.1, ymax=0.1, auto_range=False, scale='linear'):
    '''
    Create a facet plot for the given wind speed and wavelength for a certain type of error.
    '''
    with ChainedAssignent():
        # Get the data for the current wind speed
        df = df[df['wind_speed'] == wind_speed]

        # Initialize the grid
        grid = sns.FacetGrid(df, col='theta_i', hue='theta_i', palette=red_palette,
                            col_wrap=6, height=2.0)
        
        # Draw the reference line
        grid.refline(y=0, linestyle=':')

        # Draw a line plot to show the trajectory of each relative error, but space 
        # the relative error such that it is defined over the view zenith
        grid.map(sns.lineplot, 'theta_o', key, errorbar=None)

        # Draw the vertical line for solar zenith
        def add_vertical_line(data, **kwargs):
            solar_zenith = (np.rad2deg(data['solar_zenith'].iloc[0]) / 90) * 18
            plt.axvline(x=solar_zenith, color='gray', linestyle='--')

        grid.map_dataframe(add_vertical_line)

        # Adjust the tick positions and labels
        grid.set(xticks=np.arange(0, 19, 6),
                xlim=(0, 18),
                ylim=(ymin, ymax))
        
        if auto_range:
            # Custom function to set individual y-axis limits
            def set_ylim(data, **kwargs):
                plt.ylim(data[key].min(), data[key].max())
            grid.map_dataframe(set_ylim)
        else:
            grid.set(ylim=(ymin, ymax))

        # Adjust the arrangement of the plots
        grid.fig.tight_layout(w_pad=1)

        # Set the scale of the plot
        grid.set(yscale=scale)

        # Set the title of the plot
        grid.fig.suptitle(f'{name} - {wind_speed} m/s, {wavelength} μm', y=1.05)

        # Extract the legend from the first subplot and add it to the whole figure
        handles, _ = grid.axes[0].get_legend_handles_labels()
        tags = ['Solar Zenith', tag]
        grid.fig.legend(handles, tags, loc='upper right', bbox_to_anchor=(1.1, 0.9))

        # Make directory for plots/error/wavelength/key
        if not os.path.exists(f'plots/error/{str(wavelength)}/{key}'):
            os.makedirs(f'plots/error/{str(wavelength)}/{key}')        

        # Save the plot
        grid.savefig(f'plots/error/{str(wavelength)}/{key}/{key}_{wind_speed}ms_{wavelength}.png')

        # Close the plot
        plt.close(grid.fig)


def create_multi_facet_plot(name, wavelength, wind_speed, df, keys, palettes, styles, tags, ymin=-0.1, ymax=0.1, auto_range=False, scale='linear'):
    '''
    Create a multi-facet plot for the given wind speed and wavelength for multiple error types.
    '''
    with ChainedAssignent():
        # Get the data for the current wind speed
        df = df[df['wind_speed'] == wind_speed]

        # Reverse key, palette and styles order
        keys = keys[::-1]
        palettes = palettes[::-1]
        styles = styles[::-1]

        # Initialize the grid
        grid = sns.FacetGrid(df, col='theta_i', hue='theta_i',
                            col_wrap=6, height=2.0)
        
        # Draw the reference line
        grid.refline(y=0, linestyle=':')

        # Draw a line plot to show the trajectory of each key
        for key, palette, style in zip(keys, palettes, styles):
            grid.map_dataframe(sns.lineplot, x='theta_o', y=key, color=palette[0], linestyle=style)            

        # Draw the vertical line for solar zenith
        def add_vertical_line(data, **kwargs):
            solar_zenith = (np.rad2deg(data['solar_zenith'].iloc[0]) / 90) * 18
            plt.axvline(x=solar_zenith, color='gray', linestyle='--')

        grid.map_dataframe(add_vertical_line)

        # Adjust the tick positions and labels
        grid.set(xticks=np.arange(0, 19, 6),
                xlim=(0, 18))
        
        if auto_range:
            # Custom function to set individual y-axis limits
            def set_ylim(data, **kwargs):
                plt.ylim(data[keys].min().min(), data[keys].max().max())

            grid.map_dataframe(set_ylim)
        else:
            grid.set(ylim=(ymin, ymax))

        # Adjust the arrangement of the plots
        grid.fig.tight_layout(w_pad=1)

        # Set the scale of the plot
        grid.set(yscale=scale)

        # Adjust the arrangement of the plots
        grid.fig.tight_layout(w_pad=1)

        # Adjust space to make room for the title
        grid.fig.subplots_adjust(top=0.9)

        # Set the title of the plot
        grid.fig.suptitle(f'{name} - {wind_speed} m/s, {wavelength} μm', y=1.05)

        # Extract the legend from the first subplot and add it to the whole figure
        handles, _ = grid.axes[0].get_legend_handles_labels()
        grid.fig.legend(handles, tags, loc='upper right', bbox_to_anchor=(1.1, 0.9))

        # Make directory for plots/error/wavelength/key
        if not os.path.exists(f'plots/error/{str(wavelength)}/{name}'):
            os.makedirs(f'plots/error/{str(wavelength)}/{name}')   

        # Save the plot
        grid.savefig(f'plots/error/{str(wavelength)}/{name}/{name}_{wind_speed}ms_{wavelength}.png')

        # Close the plot
        plt.close(grid.fig)

def run_complete_benchmark(wavelength):
    '''
    Run the complete benchmark for the given wavelength.
    '''
    # Use all components
    components = COMPONENTS

    # Load the data
    df, benchmark_data, settings = load_data(wavelength)

    # Perform the benchmark
    benchmark(df, benchmark_data, settings, wavelength, components)

    # Process solar/viewing angles
    process_angles(df)

    # Process the errors
    process_errors(df, components)

    # Process the contributions
    process_contributions(df)

    print(df)

    # Create the facet plot for each wind speed
    progress = tqdm(wind_speeds, desc='Creating Facet Plots')
    for wind_speed in wind_speeds:
        for component in components:
            component_name = convert_channel(component)

            # Regular error plots
            create_facet_plot(f'{component_name} - Absolute Error', wavelength, wind_speed, df, f'{component_name}_abs_error', 'abs. error', auto_range=True)
            create_facet_plot(f'{component_name} - Relative Error', wavelength, wind_speed, df, f'{component_name}_rel_error', 'rel. error', -1, 1)
            
        # Percentage error plots
        create_facet_plot('Reflectance - Absolute Error (%)', wavelength, wind_speed, df, f'reflectance_abs_error_percentage', 'abs. error (%)', 0, 50)
        create_facet_plot('Reflectance - Relative Error (%)', wavelength, wind_speed, df, f'reflectance_rel_error_percentage', 'rel. error (%)', -50, 50)

        # Comparison plot
        create_multi_facet_plot("Reflectance Comparison", 
                                wavelength, wind_speed, df, 
                                keys=['reflectance', 'mitsuba_reflectance'], 
                                palettes=[green_palette, blue_palette], 
                                styles=['-', '--'], 
                                tags=['Solar Zenith', '6S Ref.', 'Mitsuba Ref.'], 
                                ymin=0, ymax=1)

        # Component value and contribution plots
        create_multi_facet_plot('Components (Mitsuba)',
                                wavelength, wind_speed, df, 
                                keys=['mitsuba_foam', 'mitsuba_glint', 'mitsuba_underlight'],
                                palettes=[red_palette, green_palette, blue_palette],
                                styles=['-', '-', '-'],
                                tags=['Solar Zenith', 'Foam', 'Glint', 'Underlight'],
                                auto_range=True)
        
        create_multi_facet_plot('Component Contribution (Mitsuba)',
                                wavelength, wind_speed, df, 
                                keys=['mitsuba_foam_contrib', 'mitsuba_glint_contrib', 'mitsuba_underlight_contrib'],
                                palettes=[red_palette, green_palette, blue_palette],
                                styles=['-', '-', '-'],
                                tags=['Solar Zenith', 'Foam', 'Glint', 'Underlight'],
                                ymin=0, ymax=1)

        progress.update(1)
    progress.close()


# Run the benchmark
if __name__ == '__main__':
    if not os.path.exists(f'plots/error'):
        os.makedirs(f'plots/error')

    # Run the benchmark for each wavelength
    total_progress = tqdm(wavelengths, desc='Running Benchmark')
    for wavelength in wavelengths:
        print(f'Running benchmark for wavelength: {wavelength}')
        run_complete_benchmark(wavelength)
        total_progress.update(1)