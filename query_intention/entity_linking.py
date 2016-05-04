import pickle, os, argparse

from sklearn.linear_model import Perceptron
from nltk.stem.wordnet import WordNetLemmatizer

from query_intention.fb_query import *
from query_instantiate import Instance, parse_to_phrases, object_decoder, get_db_entities

CANDIDATES = "candidates"
GOLD = "gold_entities"
FEATURES = "candidates_features"
REL_CANDIDATES = "rel_candidates"
LABELS = "ent_labels"

def obtain_feature(phrase, candidates):
    f = {}
    for c in candidates:
        feature = []
        ph = phrase[0][:-1]
        s = c['name']
        name = s.replace(' ','_')
        ph_id = ph.replace(' ','_').lower()
        c_id = c['id'][4:]
        if ph == s:
            feature.append("eq_"+name+"_"+ph)
        elif s.startswith(ph):
            feature.append("pre_"+name+"_"+ph)
        elif s.endswith(ph):
            feature.append("suf_"+name+"_"+ph)
        elif ph in s:
            feature.append("in_"+name+"_"+ph)
        overlap = len(s) - len(s.replace(ph, ''))
        if overlap > 0:
            feature.append("over_"+name+"_"+ph+"_"+str(overlap))

        if ph_id == c_id:
            feature.append("eq_id_"+name+"_"+ph_id)
        elif c_id.startswith(ph_id):
            feature.append("pre_id_"+name+"_"+ph_id)
        elif c_id.endswith(ph_id):
            feature.append("suf_id_"+name+"_"+ph_id)
        elif ph_id in c_id:
            feature.append("in_id_"+name+"_"+ph_id)
        overlap = len(c_id) - len(c_id.replace(ph_id, ''))
        if overlap > 0:
            feature.append("over_id_"+name+"_"+ph_id+"_"+str(overlap))
        # feature.append(c['score'])

        f[c['mid']] = feature
    return f

def get_feature(phrase, candidates):
    feature = [0] * 55
    for i in range(len(candidates)):
        c = candidates[i]
        ph = phrase[0][:-1]
        s = c['name']

        feature[i] = c['score']

        ph_id = ph.replace(' ','_').lower()

        if ph == s:
            feature[len(candidates) + i] = 1
        elif s.startswith(ph):
            feature[len(candidates)*2 + i] = 1
        elif s.endswith(ph):
            feature[len(candidates)*3 + i] = 1
        elif ph in s:
            feature[len(candidates)*4 + i] = 1
        overlap = len(s) - len(s.replace(ph, ''))
        if overlap > 0:
            feature[len(candidates)*5 + i] = overlap

        if 'id' in c.keys():
            c_id = c['id'][4:]
            if ph_id == c_id:
                feature[len(candidates)*6 + i] = 1
            elif c_id.startswith(ph_id):
                feature[len(candidates)*7 + i] = 1
            elif c_id.endswith(ph_id):
                feature[len(candidates)*8 + i] = 1
            elif ph_id in c_id:
                feature[len(candidates)*9 + i] = 1
            overlap = len(c_id) - len(c_id.replace(ph_id, ''))
            if overlap > 0:
                feature[len(candidates)*10 + i] = 1
    return feature

def get_features(phrases, candidates):
    features = []
    for i in range(len(phrases)):
        if candidates[i]:
            P = [p for p in phrases[i] if p[1] == 0]
            for j in range(len(P)):
                features.append(get_feature(P[j], candidates[i][j]))

    return features

def obtain_features(phrases, candidates):
    features = {}
    for i in range(len(phrases)):
        for j in range(len(phrases[i])):
            features.update(obtain_feature(phrases[i][-1],candidates[i][j]))
    features['x'] = []
    return features

def obtain_examples(phrases, candidates, dags, goldname):
    gold = pickle.load(open(goldname))
    instances = []
    for i in range(len(gold)):
        instances.append(Instance(phrases[i], candidates[i], dags[i], gold[i]))
    return instances


def obtain_entity_candidates(phrases, size):
    result = []

    for sentence in phrases:
        sent = []
        for phrase in sentence:
            if phrase[1] == 0:
                cand = query_freebase_entity(phrase[0], 'freebase', size)
                sent.append(cand)
        result.append(sent)
    return result

def obtain_rel_candidates(candidates, labels):
    result = []
    for i in range(len(candidates)):
        c = candidates[i]
        if len(c) == 1:
            result.append(query_freebase_property(c[0][labels[i]]['mid']))
        else:
            result.append([])
    return result

def obtain_entity_labels(candidates, entities):
    labels = []
    for j in range(len(candidates)):
        candidate = candidates[j]
        entity = entities[j]
        E = ['/' + f.replace('.', '/') for f in entity if f[:2] == 'm.' or f[:3] == 'en.']
        if len(candidate) > 0:
            for c in candidate:
                label = -1
                for i in range(len(c)):
                    C = c[i]
                    for e in E:
                        if C['mid'] == e:
                            label = i
                            break
                        if 'id' in C.keys():
                            if C['id'] == e:
                                label = i
                            break
                labels.append(label)
    return labels

def obtain_pop_score(entities):
    scoring = ['freebase','schema','schema','freebase']
    category = [False, True, True,False]
    scores = {}
    for entity in entities:
        for e in entity:
            i = e[1]
            id = e[0].lower().replace(' ','_')[:-1]
            f = query_freebase_entity(e[0], scoring[i], 10)
            if f is None:
                scores['en.' + id] = 0
            else:
                scores['en.' + id] = f['score']
    return scores

def lemmatize_word(text):
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(t,'v') for t in text]

if __name__ == "__main__":
    sep = os.path.sep
    parser = argparse.ArgumentParser(description="Obtain entity or relation candidates from Freebase")
    parser.add_argument("fpath", help="Path to features and labels (array format)", type=str)
    parser.add_argument("--size", help="Size of dataset", type=int, default=641)
    parser.add_argument("n_cand", help="Number of candidates extracted", type=int, default=641)
    parser.add_argument("type", help="Operating mode for script", type=str)
    parser.add_argument("mode", help="Training or testing split", type=str)
    args = parser.parse_args()
    path = args.fpath
    n_cand = args.n_cand
    mode = args.mode
    type = args.type
    size = args.size

    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    labels = pickle.load(open(path +"data" + sep + "questions_" + mode + "_" + str(size) + ".pickle"))
    phrases = parse_to_phrases(questions, labels)

    if 'e' in type:
        candidates = obtain_entity_candidates(phrases, n_cand)
        pickle.dump(candidates,open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'r' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle"))
        rel_candidates = obtain_rel_candidates(candidates, labels)
        pickle.dump(candidates,open(path + "data" + sep + REL_CANDIDATES + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'g' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        # rel = pickle.load(open(path + "data" + sep + REL_CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        ent, s = get_db_entities(questions)
        labels = obtain_entity_labels(candidates, ent)
        pickle.dump(labels, open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'f' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle"))
        features = get_features(phrases,candidates)
        F = [features[i] for i in range(len(features)) if labels[i] >= 0]
        L = [label for label in labels if label >= 0]
        pickle.dump(F,open(path + "data" + sep + FEATURES + "_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(L,open(path + "data" + sep + LABELS + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 't' in type:
        perc = Perceptron(n_iter=100, verbose=0)
        y = pickle.load(open(path + "data" + sep + LABELS + "_trn_641.pickle"))
        X = pickle.load(open(path + "data" + sep + FEATURES + "_trn_641.pickle"))
        w = perc.fit(X,y)
        y_tst = pickle.load(open(path + "data" + sep + LABELS + "_tst_276.pickle"))
        X_tst = pickle.load(open(path + "data" + sep + FEATURES + "_tst_276.pickle"))
        print perc.score(X_tst,y_tst)