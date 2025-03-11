import colorsys

import colour
import numpy as np


def enhance_brightness_hsv(RGB, value_increase=0.25):
    HSV = colorsys.rgb_to_hsv(RGB[0], RGB[1], RGB[2])
    HSV_brightened = (HSV[0], HSV[1], min(HSV[2] + value_increase, 1))
    return colorsys.hsv_to_rgb(HSV_brightened[0], HSV_brightened[1], HSV_brightened[2])

def compute_rgb_for_parametrization(rrs, wavelengths):
    wavelengths_5nm   =     np.arange(min(wavelengths),  max(wavelengths), 5)
    Rrs     = np.squeeze(rrs)
    Rrs_5nm = np.interp(wavelengths_5nm, wavelengths, Rrs)
    XYZ     = colour.sd_to_XYZ(colour.SpectralDistribution(dict(zip(wavelengths_5nm, Rrs_5nm)), name="Rrs_5nm"))
    RGB     = enhance_brightness_hsv(colour.XYZ_to_sRGB(XYZ / 100))
    return np.clip(RGB, 0, 1)