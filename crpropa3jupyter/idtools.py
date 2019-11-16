import datetime
import hashlib
import os
import numpy as np
import glob

from crpropa import ParticleCollector

def generateID(kwargs, filename=True, path="", compatibility = False) -> str:
    """ Generate a unique ID based on dict and optionally save it to a file """
    
    rows = ["{}={}||{}".format(str(k), v, type(v).__name__) for k, v in kwargs.items() if k not in ["UTCtime", "id"]]
    record = "\n".join(rows)

    if compatibility:
        record += "\n" # compatibility with old generateID

    id = hashlib.md5(record.encode("utf-8")).hexdigest()
    
    if not compatibility:
        record += "\n" # compatibility with old generateID

    datestamp = datetime.datetime.utcnow()
    extended_record = record + "UTCtime={}||datetime".format(
        datestamp.strftime("%Y-%m-%d %H:%M:%S")
    )

    if filename:
        with open(os.path.join(path, id + ".meta"), "w") as f:
            f.write(extended_record)
    
    return id

def readMeta(id, path="") -> dict:
    """ Read meta file """

    def readDate(datetimestring):
        return datetime.datetime.strptime(datetimestring, "%Y-%m-%d %H:%M:%S")

    types = {
        "int": int,
        "float": float,
        "float64": float,
        "str": str,
        "bool": bool,
        "datetime": readDate,
    }

    data = {}

    with open(os.path.join(path, id + ".meta"), "r") as f:
        lines = f.readlines()
        for l in lines:
            keyvalue, vtype = l.split("||")
            vtype = vtype.strip()
            pieces = keyvalue.split("=")
            key = pieces[0]
            value = "=".join(pieces[1:]) # just in case if there is an additional "=" in value

            if vtype in types.keys():
                data[key] = types[vtype](value)
            else:
                data[key] = value

    return data

def saveData(id, data, path=""):
    """ Save numpy data by ID """
    np.savez_compressed(os.path.join(path, id + ".npz"), **data)

def loadData(id, path=""):
    """ Load numpy data by ID """
    return np.load(os.path.join(path, id + ".npz"))

def isMetaFilePresent(kwargs, path=""):
    """ Read meta by ID """
    id = generateID(kwargs, filename=False, path="")
    return (readMeta(os.path.join(path, id + ".meta")) != {})

def saveXYPointsToFile(filename, xs, ys):
    """ Quickly save (x,y) data """
    with open(filename, "w") as f:
        xs_sorted, ys_sorted = [
            list(x) for x in zip(*sorted(zip(xs, ys), key=lambda pair: pair[0]))
        ]
        for x, y in zip(xs_sorted, ys_sorted):
            f.write("{:g}\t{:g}\n".format(x, y))

def loadMetaFiles(dir_path):
    """ List meta files and their content from a directory """
    list_meta = []
    for filename in glob.glob(dir_path + '/*.meta'):
        name = filename.split('/')[-1]
        id = name.split(".")[0]
        meta = readMeta(id, dir_path)
        meta["id"] = id
        list_meta.append(meta)
    return list_meta

def getDatetimeOfID(kwargs, path=""):
    id = generateID(kwargs, filename=False, path="")
    return readMeta(id+'.meta')["datetime"]

def filterRunIDs(filter_function, dir_path):
    filtered_runs = []
    for meta in loadMetaFiles(dir_path):
        if filter_function(meta):
            filtered_runs.append(meta['id'])
    return filtered_runs

