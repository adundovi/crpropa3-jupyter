from crpropa import *
import numpy as np

import crpropa3jupyter.diffusion as diffusion

def getInitialDistribution(N: "particle number", E: "energy in J", boxsize: "boxsize in m"):
    """Returns ParticleCollector container with homogeneous
    distribution of N particles of energy E within a box of size boxsize"""
    source = Source()
    source.add(SourceEnergy(E))
    source.add(SourceIsotropicEmission())
    source.add(SourceParticleType(1*nucleusId(1,1)))
    source.add(SourceUniformBox(Vector3d(-boxsize/2), Vector3d(boxsize)))

    initialDistribution = ParticleCollector()

    for i in range(N):
        c = source.getCandidate()
        initialDistribution.process(c)

    return initialDistribution

def run(collector, minDist, maxDist, config):
    """

    """
    intStep = config['intStep']
    propModule = config['propModule']
    Nsteps = config['Nsteps']
    mfield = config['mfield']

    sim = ModuleList()
    minStep = maxStep = intStep
    prop = propModule(mfield, 1e-4, minStep, maxStep)
    sim.add(prop)

    checkpoints = np.linspace(minDist, maxDist, Nsteps)

    accumulators = []
    for dist in checkpoints:
        accumulator = {
            "dist": dist,
            "output": ParticleCollector()
        }
        accumulator["output"].setClone(True)
        accumulators.append(accumulator)

        detector = DetectionLength(dist)
        detector.onReject(accumulator["output"])
        detector.setMakeRejectedInactive(False)
        sim.add(detector)
            
    drop_condition = MaximumTrajectoryLength(maxDist)
    sim.add(drop_condition)

    sim.setShowProgress()
    sim.run(collector.getContainer())
        
    return accumulators

def converge_run(simconfig):
    
    def converge_run_iteration(input_collector, minDist, maxDist):
        accumulators = run(input_collector, minDist, maxDist, simconfig)
        N = len(accumulators)
        first_half = accumulators[5:int(N/2)]
        second_half = accumulators[int(N/2):]
        last_accumulator = accumulators[-1]
        diff_of_first_half = diffusion.calc_diffusion_from_accumulators(first_half)
        diff_of_second_half = diffusion.calc_diffusion_from_accumulators(second_half)
        
        meanD_of_first_half = {}
        meanD_of_second_half = {}
        stdD_of_second_half = {}
        for axis in ['x', 'y', 'z']:
            meanD_of_first_half[axis] = np.mean(diff_of_first_half[axis])
            meanD_of_second_half[axis] = np.mean(diff_of_second_half[axis])
            stdD_of_second_half[axis] = np.std(diff_of_second_half[axis])
        
        return meanD_of_first_half, meanD_of_second_half, stdD_of_second_half, last_accumulator['output']
        
    # initial input distribution
    input_collector = getInitialDistribution(
        simconfig['Nparticles'], simconfig['Eparticle'], simconfig['boxsize'])
  
    minDist = simconfig['r_g']
    maxDist = simconfig['deltaDist']
    tolerance = simconfig['convergenceTolerance']
        
    i = 0
    while simconfig['maxIter'] > i:
        i += 1
        meanD_of_first_half, meanD_of_second_half, stdD_of_second_half, input_collector = \
            converge_run_iteration(input_collector, minDist, maxDist)
        
        minDist = i*simconfig['deltaDist']
        maxDist = (i+1)*simconfig['deltaDist']
       
        converged = [False, False, False]
        for j, axis in enumerate(['x', 'y', 'z']):
            first_second_diff = np.fabs(meanD_of_first_half[axis] - meanD_of_second_half[axis])
            if (first_second_diff < tolerance*stdD_of_second_half[axis]):
                converged[j] = True
        if all(converged):
            break
            
    if i == simconfig['maxIter']:
        convergenceProblem = True
    else:
        convergenceProblem = False
            
    return {
        'convergenceProblem': convergenceProblem,
        'D_xx': meanD_of_second_half['x'],
        'D_yy': meanD_of_second_half['y'],
        'D_zz': meanD_of_second_half['z'],
        'std_D_xx': stdD_of_second_half['x'],
        'std_D_yy': stdD_of_second_half['y'],
        'std_D_zz': stdD_of_second_half['z'],
    }
