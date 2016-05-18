"""Convert features from original string-array shape to sparse one-hot array shape"""
import argparse, pickle, numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder

def process_features(ftr):
    """Finds values of each feature for all questions

    Keyword arguments:
    ftr -- tuple of lists of features for each sample

    """
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
    """Assign each feature an integer index and convert the values of features to integers

    Keyword arguments:
    row -- one feature with values to be converted

    """
    unique = list(set(row))
    dct = dict([(item,index) for index,item in list(enumerate(unique))])
    result = [mapper(x,dct) for x in row]
    return result

def mapper(x, dct):
    """Maps feature values (strings) to integers

    Keyword arguments:
    x -- key
    dct -- dictionary of values

    """
    if x == 'x':
        return np.nan
    return dct[x]

def imputator(features):
    """Fill in missing values with mean of the remaining samples

    Keyword arguments:
    features -- feature matrix

    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(features)
    return imp.transform(features)

def encode(features, labels):
    """One-hot encode the values of each feature

    Keyword arguments:
    features -- feature matrix
    labels -- labels of samples

    """
    enc = OneHotEncoder()
    enc.fit(features)
    arr = enc.transform(features).toarray()
    result = np.array([[0 for j in range(len(arr[0])+1)] for k in range(len(arr))])
    for i in range(len(arr)):
        result[i] = np.append(arr[i], labels[i])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train weights for DAG detection")
    parser.add_argument("fpath", help="Examples file name", type=str)
    parser.add_argument("output", help="Output file name", type=str)
    args = parser.parse_args()
    path = args.fpath
    output = args.output

    examples = pickle.load(open(path))
    ftr = zip(*examples)
    features = encode(imputator(process_features(ftr[0])),ftr[1])
    np.savetxt(output + ".csv", features, delimiter=",")