from __future__ import print_function

from IPython.display import HTML

import ipyvolume as ipv
import numpy as np
import itertools
import healpy
import scipy
import pint
import csv

from crpropa import *

from .idtools import *
from .plotting import *

def loadContainer(metaOrId, meta_path=''):
    if isinstance(metaOrId, (dict,)):
        return loadDataByMeta(metaOrId, meta_path)
    else:
        return loadDataByID(metaOrId, meta_path)

def loadContainerByMeta(meta, meta_path = ''):
    output = ParticleCollector()
    output.load(meta_path + meta['id']+'.txt.gz')
    return output

def loadContainerByID(ID, meta_path=''):
    output = ParticleCollector()
    output.load(meta_path + ID+'.txt.gz')
    return output

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
        plt.plot(x,y, 'k')
    if perspective in ['XZ', 'ZX']:
        plt.plot(x,z, 'k')
    if perspective in ['YZ', 'ZY']:
        plt.plot(y,z, 'k')
    
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


