# Adapted from https://github.com/astropy/astroscrappy
# (astroscrappy/tests/fake_data.py, licensed under BSD-3)

import os

import numpy as np
from astropy.nddata import CCDData, VarianceUncertainty


# Make a simple Gaussian function for testing purposes
def gaussian(image_shape, x0, y0, brightness, fwhm):
    x = np.arange(image_shape[1])
    y = np.arange(image_shape[0])
    x2d, y2d = np.meshgrid(x, y)
    sig = fwhm / 2.35482
    normfactor = brightness / 2.0 / np.pi * sig**-2.0
    exponent = -0.5 * sig**-2.0
    exponent *= (x2d - x0) ** 2.0 + (y2d - y0) ** 2.0
    return normfactor * np.exp(exponent)


def make_fake_data(
    nimg, outdir, nsources=100, ncosmics=100, shape=(2048, 2048), dtype=np.float32
):
    # Set a seed so that the tests are repeatable
    rng = np.random.default_rng(200)

    # Add some fake sources
    sources = np.zeros(shape, dtype=np.float32)
    xx = rng.uniform(low=0.0, high=shape[0], size=nsources)
    yy = rng.uniform(low=0.0, high=shape[1], size=nsources)
    brightness = rng.uniform(low=1000.0, high=30000.0, size=nsources)
    for x, y, b in zip(xx, yy, brightness):
        sources += gaussian(shape, x, y, b, 5)

    for i in range(nimg):
        # Create a simulated image to use in our tests
        imdata = np.zeros(shape, dtype=dtype)
        # Add sky and sky noise
        imdata += 200

        imdata += sources

        # Add the poisson noise
        imdata = np.float32(rng.poisson(imdata))

        # Add readnoise
        imdata += rng.normal(0.0, 10.0, size=shape)

        # Add 100 fake cosmic rays
        cr_x = rng.integers(low=5, high=shape[0] - 5, size=ncosmics)
        cr_y = rng.integers(low=5, high=shape[1] - 5, size=ncosmics)
        cr_brightnesses = rng.uniform(low=1000.0, high=30000.0, size=ncosmics)
        imdata[cr_y, cr_x] += cr_brightnesses
        imdata = imdata.astype("f4")

        # Make a mask where the detected cosmic rays should be
        # crmask = np.zeros(shape, dtype=np.bool)
        # crmask[cr_y, cr_x] = True

        ccd = CCDData(
            imdata, uncertainty=VarianceUncertainty(imdata / 10), unit="electron"
        )
        ccd.write(os.path.join(outdir, f"image-{i+1:02d}.fits"), overwrite=True)
        print(".", end="")
