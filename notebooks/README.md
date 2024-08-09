# Oceanic Reflectance Model

This repository contains supporting code for both the 6S and Mitsuba implementations of the oceanic reflectance model. This document provides further details for 

* The workings of the Mitsuba BRDF
* The folder structure
* How to use and run certain files

## Mitsuba BRDF

### Parameters

The Mitsuba BRDF has the following parameters:

* `component` - The component to be evaluated

| Component              | Value | Includes Cosine Foreshortening |
|:----------------------:|:-----:|:------------------------------:|
| Total Reflectance      | 0     | Yes                            |
| Whitecap Reflectance   | 1     | **No**                         |
| Glint Reflectance      | 2     | **No**                         |
| Underlight Reflectance | 3     | **No**                         |

* `wavelength` - The wavelength at which to evaluate the BRDF
* `wind_speed` - The wind speed over the sea (at 10m height)
* `wind_direction` - The direction of the wind, relative to the incoming direction $\phi_i$
* `chlorinity` - The chlorinity of the sea water (logical value is 19)
* `pigmentation` - The pigment concentration of the sea water
* `shininess` - The exponent used for Blinn-Phong in the importance sampling scheme

|      Parameter       |          Range           |   Unit   |
|:--------------------:|:------------------------:|:--------:|
|      Wavelength      |  $\lambda \in [0.2, 4.0]$ |    μm    |
|      Wind Speed      |  $u \in [0, 37.2455]$     |   m/s    |
|   Wind Direction     | $\phi_w \in [0, 2\pi]$    |  radian  |
|      Chlorinity      |   $C \in [0, \infty] $    | g/kg$^{-1}$ |
|     Pigmentation     |    $P \in [0.3, 30]$      | mg/m$^3$ |
|      Shininess       |   $\sigma \in [0, \infty]$ |    /     |

### Initialization

The Mitsuba BRDF can then be initialized as follows:

```python
bsdf = mi.load_dict({
    'type': 'oceanic_legacy',
    'component': 0,
    'wavelength': 0.55,
    'wind_speed': 10,
    'wind_direction': 0,
    'chlorinity': 19,
    'pigmentation': 0.3,
    'shininess': 50,
})
```

## Folder Structure

### `validation`

The `validation` folder contains all scripts and notebooks that are related to validate the Mitsuba BRDF plugin to the original 6S oceanic reflectance model. 

* `benchmark.py` - Used to benchmark the data generated from Mitsuba compared to 6S. Running this script will take all the data points collected in the `./data` directory and produce a variety of comparable plots (which will be stored in the `./plots` directory). Benchmarking includes visualizations of the following properties:
    - **Relative error** of total reflectance, as well as each individual component (whitecap, sun glint, underlight)
    - **Absolute error** of total reflectance, as well as each individual component (whitecap, sun glint, underlight)
    - **Percentage formats** of both relative and absolute errors
    - **Component contributions**, both in percentages and true values
    - **Channel comparisons** of the total, whitecap, glint and underlight reflectances

**IMPORTANT NOTICE**: While the relative error may indicate serious issues with the model, this is actually not the case. Due to the extremely small differences, the relative error explodes. To verify the validity, make sure to always take a look at *both the relative **and** absolute errors*!