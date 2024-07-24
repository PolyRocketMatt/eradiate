import datetime
import numpy as np
import json
from multiprocessing.dummy import Pool as ThreadPool  
from Py6S import *
from tqdm import tqdm

wind_azimuth = 0
salinity = 34.3
pigmentation = 0.3

# Boundary Parameters
outgoing_azimuth = 180
outgoing_zeniths = np.linspace(0, np.deg2rad(89), 90)    

incoming_azimuth = 0
incoming_zeniths = np.linspace(0, np.deg2rad(89), 90)

# All dataset wind speeds
wind_speeds = [0.1, 1, 5, 10, 20, 37]

def generate_data(wind_speed, pbar):
    # SixS Parameters
    s = SixS()
    s.wavelength = Wavelength(0.5)
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    s.ground_reflectance = GroundReflectance.HomogeneousOcean(wind_speed, wind_azimuth, salinity, pigmentation)

    # Keep a list of temporary file names
    solar_zenith_data = {}

    # Loop over tuple of outgoing azimuths and zeniths
    for incoming_zenith in incoming_zeniths:
        reflectances = {}
        for outgoing_zenith in outgoing_zeniths:
            s.geometry = Geometry.User()
            s.geometry.solar_z = np.rad2deg(incoming_zenith)
            s.geometry.solar_a = np.rad2deg(incoming_azimuth)
            s.geometry.view_z = np.rad2deg(outgoing_zenith)
            s.geometry.view_a = np.rad2deg(outgoing_azimuth)
            s.run()
            foam = s.outputs.values['water_component_foam']
            glint = s.outputs.values['water_component_glint']
            water = s.outputs.values['water_component_water']
            reflectances[outgoing_zenith] = foam + glint + water
            pbar.update(1)

        # Save dataset as JSON to file in the 'data' directory
        solar_zenith_data[incoming_zenith] = dict(reflectances)

    # Combine all data found in the 'data' directory into a single JSON file
    combined_data = {}
    for solar_zenith, data in solar_zenith_data.items():
        combined_data[solar_zenith] = data

    # Write the combined data to a file
    with open(f'data/data_{wind_speed}ms.json', 'w') as f:
        json.dump(combined_data, f)

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    pbar = tqdm(total=len(wind_speeds) * len(incoming_zeniths) * len(outgoing_zeniths))
    
    pool = ThreadPool(8)
    for wind_speed in wind_speeds:
        pool.apply_async(generate_data, args=(wind_speed, pbar))
    pool.close()
    pool.join()
    pbar.close()
    elapsed = datetime.datetime.now() - current_time
    print(f'Finished generating data for all wind speeds! Elapsed time: {elapsed}')