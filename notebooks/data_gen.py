import datetime
import numpy as np
import json
from multiprocessing.dummy import Pool as ThreadPool  
from Py6S import *
from tqdm import tqdm

wind_direction = 0
salinity = 34.3
chlorinity = 19
pigmentation = 0.3

# Boundary Parameters
outgoing_azimuth = 180
outgoing_zeniths = np.arange(0, 86, 5)    

incoming_azimuth = 0
incoming_zeniths = np.arange(0, 86, 5)

# To radian
outgoing_zeniths = np.deg2rad(outgoing_zeniths)
incoming_zeniths = np.deg2rad(incoming_zeniths)

# All dataset wind speeds
wind_speeds = [0.1, 1, 10, 37]
wavelengths = [0.4, 0.5, 0.7, 1.5, 2.2]

def generate_data(wavelength, wind_speed, pbar):
    try:
        # SixS Parameters
        s = SixS()
        s.wavelength = Wavelength(wavelength)
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
        s.ground_reflectance = GroundReflectance.HomogeneousOcean(wind_speed, wind_direction, salinity, pigmentation)

        # Keep a list of temporary file names
        solar_zenith_data = {}

        # Loop over tuple of outgoing azimuths and zeniths
        for incoming_zenith in incoming_zeniths:
            reflectances = {}
            data_index = 0
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
                total = foam + glint + water
                if (total == 0):
                    foam_percent = 0
                    glint_percent = 0
                    water_percent = 0
                else:
                    foam_percent = foam / total
                    glint_percent = glint / total
                    water_percent = water / total
                reflectances[data_index] = {
                    'outgoing_zenith': outgoing_zenith,
                    'foam': foam,
                    'glint': glint,
                    'water': water,
                    'total': total,
                    'foam_percent': foam_percent,
                    'glint_percent': glint_percent,
                    'water_percent': water_percent
                }
                data_index += 1

            # Save dataset as JSON to file in the 'data' directory
            solar_zenith_data[incoming_zenith] = dict(reflectances)
            pbar.update(1)

        json_data = {
            'wavelength': wavelength,
            'wind_speed': wind_speed,
            'solar_azimuth': incoming_azimuth,
            'view_azimuth': outgoing_azimuth,
            'wind_direction': wind_direction,
            'salinity': salinity,
            'chlorinity': chlorinity,
            'pigmentation': pigmentation,
            'data': solar_zenith_data
        }

        # Write the combined data to a file
        with open(f'data/data_{wind_speed}ms_{wavelength}.json', 'w') as f:
            json.dump(json_data, f)
    except Exception as e:
        print(f'Error: {e}')

        # Print the whole stack trace
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    pbar = tqdm(total=len(wind_speeds) * len(wavelengths) * len(incoming_zeniths))
    
    pool = ThreadPool(32)
    for wavelength in wavelengths:
        for wind_speed in wind_speeds:
            pool.apply_async(generate_data, args=(wavelength, wind_speed, pbar))
    pool.close()
    pool.join()
    pbar.close()
    elapsed = datetime.datetime.now() - current_time
    print(f'Finished generating data for all wind speeds! Elapsed time: {elapsed}')