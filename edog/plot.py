import numpy as np
import edog.tools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc

params = {
    'text.usetex': True,
    'font.family': [u'sans-serif'],
    'font.sans-serif':
    [u'sans-serif'],
    'font.size': 10,
    'font.weight': 'semibold',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 1000,
    'figure.dpi': 1000,
    'text.latex.preamble': r"\usepackage{amsmath}",
}
plt.rcParams.update(params)


class MidpointNormalize(colors.Normalize):
    """
    Source:
    https://matplotlib.org/gallery/userdemo/colormap_normalizations_custom.html

    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
