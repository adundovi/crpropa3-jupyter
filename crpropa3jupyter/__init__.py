from __future__ import print_function

from IPython.display import HTML

import numpy as np
import ipyvolume as ipv
import matplotlib
import matplotlib.pyplot as plt
import itertools
import healpy
import scipy
import pint
import csv

from .idtools import *
from .mathtools import *
from .frequent_cr_equations import *
from .plotting import *
from .simtools import *
from .turbulence import *
from .diffusion import *

#matplotlib.use('agg')
params = {
        'backend': 'wxAgg',
#        'lines.markersize' : 2,
#        'axes.labelsize': 18,
#        'font.size': 18,
#        'font.family': 'sans-serif',
#        'legend.fontsize': 18,
#        'xtick.labelsize': 18,
#        'ytick.labelsize': 18,
#        'text.usetex': True,
        'figure.figsize': (10.0, 8.0)
}
plt.rcParams.update(params)

cellhidebutton = HTML('''<script>code_show=false; function code_toggle() {
if (code_show){$('div.input').hide();} else {$('div.input').show();}code_show = !code_show}
$( document ).ready(code_toggle); </script> <form action="javascript:code_toggle()">
<input type="submit" value="Hide source cells"></form>''')

class DefaultDir(object):
    mfields = 'magnetic_fields/'
    data    = 'data/'
    img     = 'img/'

def quickly_save_fig(title):
    plt.savefig(DefaultDir.img + title+'.png', bbox_inches='tight')
    plt.savefig(DefaultDir.img + title+'.pdf', bbox_inches='tight')

