import warnings

import mitsuba as mi
import numpy as np
import xarray as xr


def bitmap_to_dataarray(bmp: "mitsuba.Bitmap", dtype="float64") -> xr.DataArray:
    """
    Format Mitsuba bitmap data as an xarray data array.

    Parameters
    ----------
    bmp : mitsuba.core.Bitmap
        Mitsuba bitmap to be converted to a data array.

    dtype : dtype
        Data type, forwarded to :func:`numpy.array`.

    Returns
    -------
    dataset : DataArray
        Bitmap data as an xarray data array.

    Raises
    ------
    ValueError
        If `bmp` has an unsupported pixel format.
    """
    img = np.array(bmp, dtype=dtype)

    if isinstance(bmp, mi.Bitmap):
        try:
            pixel_formats = {
                mi.Bitmap.PixelFormat.Y: ["Y"],
                mi.Bitmap.PixelFormat.YA: ["Y", "A"],
                mi.Bitmap.PixelFormat.RGB: ["R", "G", "B"],
                mi.Bitmap.PixelFormat.RGBA: ["R", "G", "B", "A"],
                mi.Bitmap.PixelFormat.XYZ: ["X", "Y", "Z"],
                mi.Bitmap.PixelFormat.XYZA: ["X", "Y", "Z", "A"],
            }
            channels = pixel_formats[bmp.pixel_format()]
        except KeyError:
            if bmp.pixel_format() == mi.Bitmap.PixelFormat.MultiChannel:
                channels = [f"ch{i}" for i in range(bmp.channel_count())]
            else:
                raise ValueError(
                    f"unsupported bitmap pixel format '{bmp.pixel_format()}'"
                )

    else:
        raise TypeError

    height, width = img.shape[0], img.shape[1]

    result = xr.DataArray(
        np.reshape(img, (height, width, -1)),
        dims=["y_index", "x_index", "channel"],
        coords={
            "y_index": (
                "y_index",
                range(height),
                {"long_name": "height pixel index"},
            ),
            "x_index": (
                "x_index",
                range(width),
                {"long_name": "width pixel index"},
            ),
            "channel": (
                "channel",
                channels,
                {"long_name": "film spectral channel"},
            ),
            "y": (
                "y_index",
                np.linspace(0, 1, height),
                {"long_name": "film height coordinate"},
            ),
            "x": (
                "x_index",
                np.linspace(0, 1, width),
                {"long_name": "film width coordinate"},
            ),
        },
    )

    return result


def bitmap_to_dataset(bmp: "mitsuba.Bitmap", dtype="float64") -> xr.Dataset:
    """
    Format Mitsuba bitmap data as an xarray dataset.

    Parameters
    ----------
    bmp : mitsuba.core.Bitmap
        Mitsuba bitmap to be converted to a dataset.

    dtype : dtype
        Data type, forwarded to :func:`numpy.array`.

    Returns
    -------
    dataset : DataArray
        Bitmap data as an xarray data array.
    """
    result = xr.Dataset(
        {
            layer_name: bitmap_to_dataarray(layer, dtype=dtype)
            for layer_name, layer in bmp.split()
        }
    )
    return result
