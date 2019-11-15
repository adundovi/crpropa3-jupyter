import os
import glob
from crpropa import ParticleCollector

def addTwoAccumulators(acs1, acs2):
    """ Combine outputs of two accumulators """
    total = []
    for a1, a2 in zip(acs1, acs2):
        accumulator = {"dist": a1["dist"], "output": ParticleCollector()}
        accumulator["output"].setClone(True)

        for c in a1["output"]:
            accumulator["output"].process(c)
        for c in a2["output"]:
            accumulator["output"].process(c)

        total.append(accumulator)
    return total

def sumAccumulators(list_of_acc):
    """ Join outputs of a list accumulators """
    if len(list_of_acc) > 1:
        return addTwoAccumulators(sumAccumulators(list_of_acc[1:]), list_of_acc[0])
    return list_of_acc[0]

def saveAccumulators(accumulators, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, 0o700)

    for a in accumulators:
        a["output"].dump("{}/{}.txt.gz".format(dirname, a["dist"]))

def loadAccumulators(dirname):
    accumulators = []

    files = glob.glob(dirname+"/*.txt.gz")
    for f in files:
        accumulator = {
            "dist": float(".".join(os.path.basename(f).split(".")[0:2]))*meter,
            "output": ParticleCollector()
        }
        accumulator["output"].load(f)
        accumulator["output"].setClone(True)
        accumulators.append(accumulator)

    return sorted(accumulators, key=lambda k: k['dist'])
