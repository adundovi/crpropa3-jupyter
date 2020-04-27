from __future__ import print_function

from IPython.display import HTML

import numpy as np
import mpmath
import matplotlib
import itertools
import healpy
import scipy
import pint
import csv

from crpropa import *

import crpropa3jupyter.idtools
import crpropa3jupyter.mathtools
import crpropa3jupyter.frequent_cr_equations as common
import crpropa3jupyter.plotting
import crpropa3jupyter.simtools
import crpropa3jupyter.turbulence
import crpropa3jupyter.diffusion
import crpropa3jupyter.diffsim

matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
}

plt.rcParams.update(fonts)

#params = {
#        'backend': 'wxAgg',
#}
#plt.rcParams.update(params)

cellhidebutton = HTML('''<script>code_show=false; function code_toggle() {
if (code_show){$('div.input').hide();} else {$('div.input').show();}code_show = !code_show}
$( document ).ready(code_toggle); </script> <form action="javascript:code_toggle()">
<input type="submit" value="Hide source cells"></form>''')

class DefaultDir(object):
    mfields = 'magnetic_fields/'
    data    = 'data/'
    img     = 'img/'

def quickly_save_fig(title):
    plt.savefig(DefaultDir.img + title+'.png')
    plt.savefig(DefaultDir.img + title+'.pdf')
    plt.savefig(DefaultDir.img + title+'.eps')

