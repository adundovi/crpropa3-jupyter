from __future__ import print_function

from IPython.display import HTML

import ipyvolume as ipv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import healpy
import pint
import csv

from crpropa import *

from .simulationid import *
from .plotting import *

params = {
        'backend': 'wxAgg',
        'lines.markersize' : 2,
        'axes.labelsize': 18,
        'font.size': 18,
        'font.family': 'serif',
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': True,
        'figure.figsize': (8.0, 6.0)
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

def autosavefig(title):
    plt.savefig(DefaultDir.img + title+'.png', bbox_inches='tight')
    plt.savefig(DefaultDir.img + title+'.pdf', bbox_inches='tight')

def create_proton(E = 1*EeV, position = Vector3d(0), direction = Vector3d(0,1,0)):
    c = Candidate()
    c.current.setId(nucleusId(1, 1))
    c.current.setEnergy(E)
    c.current.setPosition(position)
    c.current.setDirection(direction)
    return c

def propagate_c_in_field(c, B, minStep = 10*pc, maxStep = 100*kpc, maxLength = 5*Mpc):
    trajectory = ParticleCollector(10000, True)
    sim = ModuleList()
    prop = PropagationCK(B, 1e-4, minStep, maxStep)
    sim.add(prop)
    sim.add(MaximumTrajectoryLength(maxLength))
    sim.add(trajectory)
    sim.run(c, True)
    return trajectory

def get_points(container):
    points = []
    for c in container:
        pos = c.current.getPosition()
        points.append([pos.getX(), pos.getY(), pos.getZ()])
    return np.array(points)

def getDistanceFromSource(c):
    return (c.source.getPosition() - c.current.getPosition()).getR()

def plot_trajectory(trajectory, perspective='XY', length=1):
    points = get_points(trajectory)
    x, y, z = points[:,0], points[:,1], points[:,2]

    scale = 'meter'
    if length != meter:
        x = x/length
        y = y/length
        z = z/length
        scale = length
    elif max(x)/Mpc > 0.1:
        x = x/Mpc
        y = y/Mpc
        z = z/Mpc
        scale = 'Mpc'
    elif max(x)/kpc > 0.1:
        x = x/kpc
        y = y/kpc
        z = z/kpc
        scale = 'kpc'

    if perspective in ['XY', 'YX']:
        plt.plot(x,y)
    if perspective in ['XZ', 'ZX']:
        plt.plot(x,z)
    if perspective in ['YZ', 'ZY']:
        plt.plot(y,z)
    
    #plt.xlabel(r"$y / {{\rm {0} }}$".format(scale))
    #plt.ylabel(r"$z / {{\rm {0} }}$".format(scale))

def plot_trajectory_3d(trajectory):
    ipv.pylab.figure()
    
    points = get_points(trajectory)
    x, y, z = points[:,0], points[:,1], points[:,2]

    scale = 'meter'
    if max(x)/Mpc > 0.1:
        x = x/Mpc
        y = y/Mpc
        z = z/Mpc
        scale = 'Mpc'
    
    s = ipv.pylab.scatter(x, y, z, size=0.5, marker="sphere")
    ipv.pylab.animation_control(s)
    ipv.pylab.show()

def plot_field_lines(B, start, end, perspective = 'XY', unit = Mpc, density = 0.8):
    N_steps = 100
    N_steps_c = 100j

    start_i = int(start.getX())
    end_i = int(end.getX())
    start_j = int(start.getY())
    end_j = int(end.getY())
    z = start.getZ()

    points_x = np.linspace(start_i, end_i, N_steps)
    points_y = np.linspace(start_j, end_j, N_steps)

    Y, X = np.mgrid[start_i:end_i:N_steps_c, start_j:end_j:N_steps_c]

    U = np.zeros((N_steps, N_steps), dtype=float)
    V = np.zeros((N_steps, N_steps), dtype=float)

    for ix, x in enumerate(points_x):
        for iy, y in enumerate(points_y):
            if perspective in ['XY', 'YX']:
                U[iy][ix] = B.getField(Vector3d(x, y, z)*unit).getX()
                V[iy][ix] = B.getField(Vector3d(x, y, z)*unit).getY()
            if perspective in ['XZ', 'ZX']:
                U[iy][ix] = B.getField(Vector3d(x, z, y)*unit).getX()
                V[iy][ix] = B.getField(Vector3d(x, z, y)*unit).getZ()
            if perspective in ['YZ', 'ZY']:
                U[iy][ix] = B.getField(Vector3d(z, x, y)*unit).getY()
                V[iy][ix] = B.getField(Vector3d(z, x, y)*unit).getZ()

    plt.streamplot(X, Y, U, V, density=density,
                   arrowstyle='->', arrowsize=1.5)

def getUniformSphereVectors(N=1000):
    # put points on a spiral around the sphere
    node = {}

    s  = 3.6/np.sqrt(N)
    dz = 2.0/N
    long = 0
    z    = 1 - dz/2
    for k in np.arange(0,N):
        r = np.sqrt(1-z*z)
        node[k] = (np.cos(long)*r, np.sin(long)*r, z)
        z = z - dz
        long = long + s/r

    vectors = []
    for n in node.values():
        x, y, z = n
        v = Vector3d(x, y, z)
        vectors.append(v)
        
    return vectors


def linreg(xs, ys, debug=False):
    A = np.vstack([xs, np.ones(len(xs))]).T
    res = np.linalg.lstsq(A, ys)
    
    a, b = res[0]
    residuals = res[1]

    if debug:
        print("slope = {}, y-cut = {}".format(a,b))

    return a, b, residuals

def approx(first, second):
    if np.fabs(first - second) <= 0.01*second:
        return True
    return False

