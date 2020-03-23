import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import healpy

from crpropa import *

default_html_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def container2crmap(output, limit=0):
    
    tmp_lat = []; tmp_lon = []
    
    container = ParticleMapsContainer()
    sourceEnergyWeightExponent = 1
    
    weight = 1./output.size()
    
    i = 0
    
    for c in output:
        if not np.isfinite(c.current.getPosition().getX()):
            continue
        v = c.current.getDirection()
        
        lon = v.getPhi()
        lat = np.pi/2 - v.getTheta()
        tmp_lon.append(lat)
        tmp_lat.append(lon)
        
        if limit != 0 and i > limit:
            break
        i += 1
    
        container.addParticle(c.current.getId(), c.current.getEnergy()+np.random.uniform(1,1000), lon, lat, weight) # hack
        
    crMap = np.zeros(49152)
    for pid in container.getParticleIds():
        energies = container.getEnergies(int(pid))
        for i, energy in enumerate(energies):
            crMap += container.getMap(int(pid), energy * crpropa.eV)
    
    return crMap

def container2skymap(output, trajectoryBottomCut=0*Mpc, trajectoryTopCut=10*Gpc):
    tmp_lat = []; tmp_lon = []; traj0 = []
    for c in output:
        if c.getTrajectoryLength() < trajectoryBottomCut or c.getTrajectoryLength() > trajectoryTopCut:
            continue
        v = c.current.getDirection()
        traj0.append(c.getTrajectoryLength())
        tmp_lat.append(np.pi/2 - v.getTheta())
        tmp_lon.append(v.getPhi())


    rel_traj = np.array(traj0)/(max(traj0)-np.min(traj0))
    print("Lat. borders: ", np.min(tmp_lat)/np.pi, max(tmp_lat)/np.pi)
    print("Lon. borders: ", np.min(tmp_lon)/np.pi, max(tmp_lon)/np.pi)

    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')
    plt.scatter(tmp_lon, tmp_lat, s=30, marker='o', c=rel_traj, linewidths=0, alpha=1)
    plt.grid(True)
    plt.colorbar()
