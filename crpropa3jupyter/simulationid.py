import datetime
import hashlib

def generateID(kwargs, filename=True, path=""):
    record = ""
    for k, v in kwargs.items():
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
             "UTCtime": readDate,
            }
    with open(path + filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            key, value = l.split("=")
            if key in types.keys():
                data[key] = types[key](value)
            else:
                data[key] = value

    return data

def readID(kwargs, path=""):
    id = generateID(kwargs, filename=False, path="")
    return readMeta(id+'.meta')
