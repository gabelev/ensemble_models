import random
import time
import sys


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs


# Returns value if key exists, and " " if not.
# Takes searched key and dictionary as input.
def getit(item, store):
    if item in store:
        return store.get(item)
    else:
        return " "


#takes file name as input, returns reformated data in three lists?
def reformat_data(file_name):
    print("starting reformat_data")
    with open(file_name, "r") as infile:
        initial_data = []
        for line in infile:
            if line == "\n":
                continue
            tmp = {}
            tmp["label"] = line.split()[0]
            for item in line.split("|")[1:]:
                tmp[item[0]] = item[2:].rstrip()
            #tmp.update({item[0]: item[2:].rstrip() for item in line.split("|")[1:]})
            if tmp.get("t"):
                tmp["t"] = tmp["t"]
            initial_data.append(tmp)

    #shuffles the intial data and pulls out the label in order.
    random.shuffle(initial_data)
    labels = []
    for item in initial_data:
        label = ""
        if item["label"] == '-1.0':
            label = 0
        if item["label"] == '1.0':
            label = 1
        labels.append(label)
        del item["label"]

    with open("intial_data.txt", "w") as out_file:
        for line in initial_data:
            out_file.write(line)

    with open("label_list.txt", "w") as out_file:
        for line in labels:
            out_file.write(line)

    #takes the formated dataset and creates a new list of dictionaries with interations.
    data_feature_interaction = []
    for line in initial_data:
        temp_dict = line.copy()
        tmp = {
            "si": getit("s", line) + " " + getit("i", line),
            "pi": getit("p", line) + " " + getit("i", line),
            "mi": getit("m", line) + " " + getit("i", line),
            "ai": getit("a", line) + " " + getit("i", line),
            "ps": getit("p", line) + " " + getit("s", line),
            "ei": getit("e", line) + " " + getit("i", line),
            "ri": getit("r", line) + " " + getit("i", line),
            "pc": getit("p", line) + " " + getit("c", line),
            "pb": getit("p", line) + " " + getit("b", line),
            "bi": getit("b", line) + " " + getit("i", line),
            "ki": getit("k", line) + " " + getit("i", line),
            "pk": getit("p", line) + " " + getit("k", line),
            "wi": getit("w", line) + " " + getit("i", line),
        }
        temp_dict.update(tmp)
        data_feature_interaction.append(temp_dict)

    with open("feature_interaction_list.txt", "w") as out_file:
        for line in data_feature_interaction:
            out_file.write(line)

    return

if __name__ == "__main__":

    # argunments: file_name
    file_name = sys.argv[1]

    with Timer() as t:
        reformat_data(file_name)
    print "=> elasped time: %s s" % t.secs
