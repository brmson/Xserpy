import argparse,pickle,random
from collections import defaultdict

def compute_score(features,weights,index):
    score = 0
    for feat in features:
        if feat not in weights:
            continue
        score += weights[feat][index]
    return score

def predict(weights,features,cl):
    classes = range(cl)
    scores = defaultdict(float)
    for feat in features:
        if feat not in weights:
            continue
        weight = weights[feat]
        for c, w in weight.items():
            scores[c] += w
    return max(classes, key=lambda c: (scores[c], c))

def init_weights(examples,weights,cl):
    for e,t in examples:
        for f in e:
            if f not in weights.keys():
                weights[f] = {}
                for j in range(cl):
                    weights[f][j] = 0
    return weights

def train(n_iter, examples,weights,cl):
    learning_rate = 0.1
    for i in range(n_iter):
        err = 0
        for features, true in examples:
            guess = predict(weights,features,cl)
            if guess != true:
                for f in features:
                    weights[f][true] += learning_rate
                    weights[f][guess] -= learning_rate
                err += 1.0
        random.shuffle(examples)
    print err/len(examples)
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
    examples = zip(words,labels)
    w = train(args.n_iter,examples,init_weights(examples,{},5),5)
    # pickle.dump(w,open("w_90_"+str(args.n_iter)+".pickle","wb"))