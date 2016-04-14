import argparse, pickle, numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder

def process_features(ftr):
    features = {}
    i = 0
    for F in ftr:
        for feature in F:
            f = feature.split('.')
            if f[0] not in features.keys():
                features[f[0]] = ['x' for j in range(len(ftr))]
            features[f[0]][i] = f[1]
        i += 1
    return np.matrix([process_row(features[x]) for x in features.keys()]).T

def process_row(row):
    unique = list(set(row))
    dct = dict([(item,index) for index,item in list(enumerate(unique))])
    result = [mapper(x,dct) for x in row]
    return result

def mapper(x, dct):
    if x == 'x':
        return np.nan
    return dct[x]

def imputator(features):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(features)
    return imp.transform(features)

def encode(features, labels):
    enc = OneHotEncoder()
    enc.fit(features)
    arr = enc.transform(features).toarray()
    result = np.array([[0 for j in range(len(arr[0])+1)] for k in range(len(arr))])
    for i in range(len(arr)):
        result[i] = np.append(arr[i], labels[i])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train weights for DAG detection")
    parser.add_argument("fpath", help="Path to file", type=str)
    args = parser.parse_args()
    path = args.fpath

    examples = pickle.load(open(path))
    ftr = zip(*examples)
    features = encode(imputator(process_features(ftr[0])),ftr[1])
    np.savetxt("features.csv", features, delimiter=",")