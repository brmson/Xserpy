import argparse, pickle, random,os
from collections import defaultdict

def compute_score(features, weights, index):
    score = 0
    for feat in features:
        if feat not in weights:
            continue
        score += weights[feat][index]
    return score

def predict(weights, features, cl):
    classes = range(cl)
    scores = defaultdict(float)
    for feat in features:
        if feat not in weights:
            continue
        weight = weights[feat]
        for c,  w in weight.items():
            scores[c] += w
    return max(classes,  key=lambda c: (scores[c],  c))

def init_weights(examples, weights, cl):
    for e, t in examples:
        for f in e:
            if f not in weights.keys():
                weights[f] = {}
                for j in range(cl):
                    weights[f][j] = 0
    return weights

def train(n_iter,  examples, weights, cl):
    learning_rate = 10
    for i in range(n_iter):
        err = 0
        for features,  true in examples:
            guess = predict(weights, features, cl)
            if guess != true:
                for f in features:
                    weights[f][true] += learning_rate
                    weights[f][guess] -= learning_rate
                err += 1.0
        random.shuffle(examples)
        print err/len(examples)
    return weights

def compute_error(features, labels, weights):
    error = 0
    for i in range(len(features)):
        label = labels[i]
        guess = predict(weights, features[i],5)
        if guess != label:
            error += 1.0
    print error/len(features)

if __name__ == "__main__":
    sep = os.path.sep

    parser = argparse.ArgumentParser(description="Train weights for detecting phrases")
    parser.add_argument("fpath", help="Path to features and labels (array format)", type=str)
    parser.add_argument("n_iter", help="Number of iterations for training", type=int, default=0)
    parser.add_argument("size", help="Size of dataset", type=int, default=0)
    parser.add_argument("type", help="How examples are loaded", type=str)
    parser.add_argument("mode", help="Training or testing mode, required values: trn or tst", type=str)
    args = parser.parse_args()

    path = args.fpath
    mode = args.mode

    if 'l' in args.type:
        words = pickle.load(open(path + "data" + sep + "phrase_detect_features_" + mode + "_" + str(args.size) + "_arr.pickle"))
        labels = pickle.load(open(path + "data" + sep + "labels_" + mode + "_" + str(args.size) + ".pickle"))
        examples = zip(words, labels)
        pickle.dump(examples,open(path + "data" + sep + "phr_detect_examples_" + mode + "_" + str(args.size) + ".pickle","wb"))
    else:
        examples = pickle.load(open(path + "data" + sep + "phr_detect_examples_" + mode + "_" + str(args.size) + ".pickle"))

    w = train(args.n_iter, examples, init_weights(examples, {}, 5), 5)
    pickle.dump(w, open(path + "models" + sep + "w_" + str(args.size) + "_i" + str(args.n_iter) + ".pickle", "wb"))