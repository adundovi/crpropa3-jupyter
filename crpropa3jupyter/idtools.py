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
                if vtype == 'bool':
                    data[key] = True if value.lower() in ("yes", "true", "t", "1") else False
                else:
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
    saveDataToTabularFile(filename, "# x    y", xs, ys)

def saveDataToTabularFile(filename, header, *args):
    """ Quickly save (x,y) data """
    
    sorted_columns = [
        list(x) for x in zip(*sorted(zip(*args), key=lambda pair: pair[0]))
    ]

    with open(filename, "w") as f:
        f.write("#" + "\t".join(header) + "\n")
        formatting_string = "{:g}\t" * len(sorted_columns) + "\n"
        for row in zip(*sorted_columns):
            f.write(formatting_string.format(*row))

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

def getKeyFromID(key, run_id, datapath):
    meta = readMeta(run_id, datapath)
    return meta[key]

def filterRunIDs(filter_function, dir_path, verbose=False):
    filtered_runs = []
    for meta in loadMetaFiles(dir_path):
        if verbose:
            print("ID: ", meta['id'])
        if filter_function(meta):
            filtered_runs.append(meta['id'])
    return filtered_runs

def compareMetaData(runs, datapath):
    saved_meta = {}
    for r in runs:
        meta = readMeta(r, datapath)
        for key, value in meta.items():
            if key == 'UTCtime': # skip datetime
                continue
            if key not in saved_meta.keys():
                saved_meta[key] = [value]
                continue
            if value not in saved_meta[key]:
                saved_meta[key].append(value)
    return saved_meta
