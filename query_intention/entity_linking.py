import pickle, os, argparse

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from nltk.stem.wordnet import WordNetLemmatizer

from query_intention.fb_query import *
from query_instantiate import Instance, parse_to_phrases, parse_dags, object_decoder, get_db_entities

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

def obtain_entity_labels(candidates, entities, phrases):
    labels = []
    for j in range(len(candidates)):
        candidate = candidates[j]
        entity = entities[j]
        phrase = phrases[j]
        p = [phr[0][:-1] for phr in phrase if phr[1] == 0]
        E = ['/' + f.replace('.', '/') for f in entity if f[:2] == 'm.' or f[:3] == 'en.']
        if len(candidate) > 0:
            for k in range(len(candidate)):
                c = candidate[k]
                name = p[k]
                label = -1
                found = False
                for i in range(len(c)):
                    C = c[i]
                    for e in E:
                        if C['mid'] == e:
                            label = i
                            found = True
                            break
                        elif 'id' in C.keys():
                            if C['id'] == e:
                                label = i
                                found = True
                                break
                        elif 'name' in C.keys():
                            pass
                            if C['name'] == name:
                                label = i
                                found = True
                                break
                    if found:
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

def get_edge_features(phrase, dag, k, j):
    # k: edge target
    # j: edge start
    q_type = ['who ', 'what ', 'when ', 'where ', 'how many ', 'where ']
    q_var = [V for V in phrase if V[1] == 3][0][0]
    f6_11 = [0] * len(q_type)
    if q_var in q_type:
        f6_11[q_type.index(q_var)] = 1
    ents = len([P for P in phrase if P[1] == 0])
    f1 = len(dag[j])
    f2 = ents
    f3 = phrase[j][1]
    f4 = phrase[k][1]
    f5 = len([item for sublist in dag for item in sublist])
    return [f1, f2, f3, f4, f5] + f6_11

def edge_gold_standard(phrases, dags):
    start = 0
    result = []
    features = []
    for i in range(start, 100):# range(len(dags)):
        phrase = phrases[i]
        dag = dags[i]

        for j in range(len(dag)):
            d = dag[j]
            if len(d) > 0:
                p = phrase[j]
                print p[0]
                for k in d:
                    features.append(get_edge_features(phrase, dag, k, j))
                    inp = raw_input(phrase[k][0] + " ")
                    result.append(int(inp))
        if i % 10 == 0:
            pickle.dump(features,open("partial_features.pickle","wb"))
            pickle.dump(result,open("partial_labels.pickle","wb"))
    return result, features

def lemmatize_word(text):
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(t,'v') for t in text]

def label_edges(phrase, dag, perc, j):
    dct = ['SP', 'PO']
    result = ['x'] * len(phrase)
    for d in dag[j]:
        feature = get_edge_features(phrase, dag, d, j)
        label = perc.predict(feature)
        result[d] = dct[label]
    return result

def label_entity(phrase, perc, candidate):
    feature = get_feature(phrase, candidate)
    label = perc.predict(feature)[0]
    if label < len(candidate):
        return candidate[label]['mid'].replace('/','.')[1:]
    elif len(candidate) > 0:
        return candidate[0]['mid'].replace('/','.')[1:]
    else:
        return ''

def label_relation(phrase, rel_lr, bow_dct, g_dct):
    r_bow, v_bow = get_idx(phrase, bow_dct)
    features = construct_entity_features([r_bow],[v_bow], len(bow_dct))[0]
    label = rel_lr.predict(features)[0]
    for k in g_dct.keys():
        if label == g_dct[k]:
            return k
    return 'relation'


def label_all(phrase, dag, candidates, ent_path, ed_path, rel_path, bow_path, g_path):
    ent_perc = pickle.load(open(ent_path))
    rel_lr = pickle.load(open(rel_path))
    bow_dct = pickle.load(open(bow_path))
    g_dct = pickle.load(open(g_path))
    edge_perc = pickle.load(open(ed_path))
    result = []
    for i in range(len(phrase)):
        e = 0
        if phrase[i][1] == 3:
            result.append('?x')
        elif phrase[i][1] == 0:
            result.append(label_entity(phrase[i], ent_perc, candidates[e]))
            e += 1
        elif phrase[i][1] == 1:
            result.append(label_relation(phrase, rel_lr, bow_dct, g_dct))
        if dag[i]:
            result += label_edges(phrase, dag, edge_perc, i)
        else:
            result += ['x'] * (len(phrase))
    return result

def get_bow(phrases, ent):
    q_type = ['who ', 'what ', 'when ', 'where ', 'how many ', 'where ']
    v_bow = []
    bow_dct = {}
    g_dct = {}
    gold = []
    r_bow = []
    i = 0
    k = 0
    for j in range(len(phrases)):
        phrase = phrases[j]
        ents = ent[j]
        g = [pp for pp in ents if pp[:3] != 'en.' and pp[:2] != 'm.']
        v = [pp for pp in phrase if pp[1] == 3]
        r = [pp for pp in phrase if pp[1] == 1]
        if len(v) > 0:
            p = v[0]
            if p[0] in q_type:
                v_bow.append(q_type.index(p[0]))
            else:
                v_bow.append(len(q_type))
        else:
            v_bow.append(-1)
        if len(r) > 0:
            rr = []
            for R in r[0][0].split():
                if R not in bow_dct.keys():
                    bow_dct[R] = i
                    i += 1
                rr.append(bow_dct[R])
            r_bow.append(rr)
        for G in g:
            if G not in g_dct.keys():
                g_dct[G] = k
                k += 1
        gold.append(g_dct[g[-1]])
    return bow_dct, v_bow, r_bow, gold, g_dct

def construct_entity_features(r_bow, v_bow, length):
    features = []
    for i in range(len(r_bow)):
        feature = [0] * (length + 7)
        r = r_bow[i]
        v = v_bow[i]
        if v >= 0:
            feature[v] = 1
        for R in r:
            feature[R + 6] = 1
        features.append(feature)
    return features

def get_idx(phrase, bow_dct):
    q_type = ['who ', 'what ', 'when ', 'where ', 'how many ', 'where ']
    v = [pp for pp in phrase if pp[1] == 3]
    r = [pp for pp in phrase if pp[1] == 1]
    if len(r) > 0:
        rr = []
        for R in r[0][0].split():
            if R in bow_dct.keys():
                rr.append(bow_dct[R])
    if len(v) > 0:
        p = v[0]
        if p[0] in q_type:
            vv = q_type.index(p[0])
        else:
            vv = len(q_type)
    else:
        vv = -1
    return rr, vv

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
        ent, s = get_db_entities(questions)
        bow_dct, v_bow, r_bow, y, g_dct = get_bow(phrases, ent)
        length = len(bow_dct.keys())
        X = construct_entity_features(r_bow, v_bow, length)
        csf = OneVsRestClassifier(LogisticRegression(verbose=0, max_iter=100))
        csf.fit(X, y)
        # candidates = obtain_entity_candidates(phrases, n_cand)
        pickle.dump(csf,open("relation_lr_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(bow_dct,open("bow_dict_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(g_dct,open("gold_dict_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'r' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle"))
        rel_candidates = obtain_rel_candidates(candidates, labels)
        pickle.dump(candidates,open(path + "data" + sep + REL_CANDIDATES + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'g' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        ent, s = get_db_entities(questions)
        labels = obtain_entity_labels(candidates, ent, phrases)
        # print labels.count(-1)
        pickle.dump(labels, open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'f' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle"))
        features = get_features(phrases,candidates)
        F = [features[i] for i in range(len(features)) if labels[i] >= 0]
        L = [label for label in labels if label >= 0]
        pickle.dump(F,open(path + "data" + sep + FEATURES + "_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(L,open(path + "data" + sep + LABELS + "_" + mode + "_" + str(size) + ".pickle","wb"))

    if 'l' in type:
        dags = parse_dags(phrases)
        e = edge_gold_standard(phrases, dags)

    if 't' in type:
        perc = OneVsRestClassifier(LogisticRegression(verbose=0, max_iter=50)) # Perceptron(n_iter=50, verbose=0)
        y = pickle.load(open(path + "data" + sep + LABELS + "_trn_641.pickle"))
        X = pickle.load(open(path + "data" + sep + FEATURES + "_trn_641.pickle"))
        w = perc.fit(X,y)
        pickle.dump(w, open("ent_lr_trn_641.pickle","wb"))
        y_tst = pickle.load(open(path + "data" + sep + LABELS + "_tst_276.pickle"))
        X_tst = pickle.load(open(path + "data" + sep + FEATURES + "_tst_276.pickle"))
        print perc.score(X_tst,y_tst)

    if 'd' in type:
        perc = Perceptron(n_iter=100, verbose=0)
        y = pickle.load(open(path + "query_intention" + sep + "edge_labels_100.pickle"))
        X = pickle.load(open(path + "query_intention" + sep + "edge_features_100.pickle"))
        w = perc.fit(X,y)
        pickle.dump(w, open("edge_perceptron_trn_100.pickle","wb"))
        y_tst = pickle.load(open(path + "query_intention" + sep + "edge_labels_tst_100.pickle"))
        X_tst = pickle.load(open(path + "query_intention" + sep + "edge_features_tst_100.pickle"))
        print perc.score(X_tst,y_tst)

    if 'a' in type:
        dags = parse_dags(phrases)
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        mapped = []
        for i in range(len(phrases)):
            mapped.append(label_all(phrases[i], dags[i], candidates[i], "ent_perceptron_trn_641.pickle", "edge_perceptron_trn_100.pickle"))
        pickle.dump(mapped, open("emapped.pickle","wb"))