# From https://github.com/astropy/astroscrappy/blob/master/astroscrappy/tests/fake_data.py

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
    exponent *= (x2d - x0)**2.0 + (y2d - y0)**2.0
    return normfactor * np.exp(exponent)


def make_fake_data(nimg,
                   outdir,
                   nsources=100,
                   shape=(2048, 2048),
                   dtype=np.float32):
    # Set a seed so that the tests are repeatable
    np.random.seed(200)

    # Add some fake sources
    sources = np.zeros(shape, dtype=np.float32)
    xx = np.random.uniform(low=0.0, high=shape[0], size=nsources)
    yy = np.random.uniform(low=0.0, high=shape[1], size=nsources)
    brightness = np.random.uniform(low=1000., high=30000., size=nsources)
    for x, y, b in zip(xx, yy, brightness):
        sources += gaussian(shape, x, y, b, 5)

    for i in range(nimg):
        # Create a simulated image to use in our tests
        imdata = np.zeros(shape, dtype=dtype)
        # Add sky and sky noise
        imdata += 200

        imdata += sources

        # Add the poisson noise
        imdata = np.float32(np.random.poisson(imdata))

        # Add readnoise
        imdata += np.random.normal(0.0, 10.0, size=shape)

        # Add 100 fake cosmic rays
        cr_x = np.random.randint(low=5, high=shape[0] - 5, size=100)
        cr_y = np.random.randint(low=5, high=shape[1] - 5, size=100)
        cr_brightnesses = np.random.uniform(low=1000.0, high=30000.0, size=100)
        imdata[cr_y, cr_x] += cr_brightnesses
        imdata = imdata.astype('f4')

        # Make a mask where the detected cosmic rays should be
        # crmask = np.zeros(shape, dtype=np.bool)
        # crmask[cr_y, cr_x] = True

        ccd = CCDData(imdata,
                      uncertainty=VarianceUncertainty(imdata / 10),
                      unit="electron")
        ccd.write(os.path.join(outdir, f'image-{i+1:02d}.fits'),
                  overwrite=True)
        print('.', end='')
