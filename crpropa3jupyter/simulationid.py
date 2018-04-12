import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import healpy
import datetime
import hashlib
import glob

from crpropa import *



def generateID(kwargs, filename=True, path=""):
    record = ""
    for k, v in kwargs.items():
        if k in ["UTCtime", "id"]:
            continue
        record += "{}={}\n".format(str(k),v)
    id = hashlib.md5(record.encode('utf-8')).hexdigest()

    datestamp = datetime.datetime.utcnow()
    extended_record = record+"UTCtime={}".format(
        datestamp.strftime("%Y-%m-%d %H:%M:%S")
    )

    if filename:
        with open(path + id+'.meta', 'w') as f:
            f.write(extended_record)
    return id

def readMeta(filename, path = ""):

    def readDate(datetimestring):
        return datetime.datetime.strptime(datetimestring,'%Y-%m-%d %H:%M:%S')

    data = {}
    types = {
             "N": int,
             "E": float,
             "Z": int,
             "Brms": float,
             "Lmin": float,
             "Lmax": float,
             "Lc": float,
             "n": float,
             "alpha": float,
             "dist": float,
             "note": str,
             "id": str,
             "UTCtime": readDate,
            }
    with open(path + filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            key, value = l.split("=")
            if "UTCtime" in key:
                continue
            if key in types.keys():
                data[key] = types[key](value)
            else:
                data[key] = value

    return data

def loadMetaFiles(directory):
    list_meta = []
    for filename in glob.glob(directory):
        meta = readMeta(filename)
        name = filename.split('/')[-1]
        meta["id"] = name.split(".")[0]
        list_meta.append(meta)
    return list_meta

def readID(kwargs, path=""):
    id = generateID(kwargs, filename=False, path="")
    return readMeta(id+'.meta')

def loadData(metaOrId, meta_path=''):
    if isinstance(metaOrId, (dict,)):
        return loadDataByMeta(metaOrId, meta_path)
    else:
        return loadDataByID(metaOrId, meta_path)

def loadDataByMeta(meta, meta_path = ''):
    output = ParticleCollector()
    output.load(meta_path + meta['id']+'txt.gz')
    return output

def loadDataByID(ID, meta_path=''):
    output = ParticleCollector()
    output.load(meta_path + ID+'.txt.gz')
    return output
