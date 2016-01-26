import argparse,pickle,random
from collections import defaultdict

def find_argmax(w,candidates):
    L = [sum([a*b for a,b in zip(w,candidate)]) for candidate in candidates]
    return candidates[L.index(max(L))]

def predict(weights,features):
    classes = range(5)
    scores = defaultdict(float)
    for feat in features:
        if feat not in weights:
            continue
        weights = weights[feat]
        for c, weight in weights.items():
            scores[c] += weight
    return max(classes, key=lambda c: (scores[c], c))


def train(n_iter, examples):
    learning_rate = 0.1
    weights = {}
    for e,t in examples:
        for f in e:
            weights[f] = {}
            for j in range(5):
                weights[f][j] = 0
    for i in range(n_iter):
        err = 0
        for features, true in examples:
            guess = predict(weights,features)
            if guess != true:
                for f in features:
                    weights[f][true] += learning_rate
                    weights[f][guess] -= learning_rate
                err += 1.0
        print err/len(examples)
        random.shuffle(examples)
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("n_iter",help="iterations",type=int,default=0)
    args = parser.parse_args()
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    words = pickle.load(open(path+"annotate\\phrase_detect_features_90_arr.pickle"))
    labels = pickle.load(open(path+"data\\labels_trn_90.pickle"))
    # print len(words),len(labels)
    # n = len(words[0])
    w = train(args.n_iter,zip(words,labels))
    pickle.dump(w,open("w_90_"+str(args.n_iter)+".pickle","wb"))