"""Link a DAG to knowledge base"""
import pickle, os, argparse

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import product

from query_intention.fb_query import *
from query_instantiate import Instance, parse_to_phrases, parse_dags, object_decoder, get_db_entities

CANDIDATES = "candidates"
GOLD = "gold_entities"
FEATURES = "candidates_features"
REL_CANDIDATES = "rel_candidates"
LABELS = "ent_labels"

def obtain_feature(phrase, candidates):
    """Construct a feature vector for a phrase; Unused

    Keyword arguments:
    phrase -- entity phrase
    candidates -- possible classes to link the phrase to

    """
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
        feature.append(c['score'])

        f[c['mid']] = feature
    return f

def get_feature(phrase, candidates):
    """Construct a feature vector for a phrase

    Keyword arguments:
    phrase -- entity phrase
    candidates -- list of possible classes to link the phrase to

    """
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
    """Construct a feature vector for all phrases

    Keyword arguments:
    phrases -- list of entity phrase
    candidates -- list of lists of possible classes to link the phrase to

    """
    features = []
    for i in range(len(phrases)):
        if candidates[i]:
            P = [p for p in phrases[i] if p[1] == 0]
            for j in range(len(P)):
                features.append(get_feature(P[j], candidates[i][j]))

    return features

def obtain_features(phrases, candidates):
    """Construct a feature vector for all phrases; Unused

    Keyword arguments:
    phrases -- list of entity phrases
    candidates -- list of lists of possible classes to link the phrase to

    """
    features = {}
    for i in range(len(phrases)):
        for j in range(len(phrases[i])):
            features.update(obtain_feature(phrases[i][-1],candidates[i][j]))
    features['x'] = []
    return features

def obtain_entity_candidates(phrases, size):
    """Obtain candidates for all detected entities from Freebase

    Keyword arguments:
    phrases -- list of entity phrases
    size -- how many candidates should be kept

    """
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
    """Obtain candidates for all detected relations from Freebase

    Keyword arguments:
    candidates -- candidates for entity phrases
    labels -- list of indexes

    """
    result = []
    for i in range(len(candidates)):
        c = candidates[i]
        if len(c) == 1:
            result.append(query_freebase_property(c[0][labels[i]]['mid']))
        else:
            result.append([])
    return result

def obtain_entity_labels(candidates, entities, phrases):
    """Compare the candidates from Freebase to correct entities extracted from logic formula and form gold standard

    Keyword arguments:
    candidates -- candidates for entity phrases
    entities -- list of entities detected in dataset
    phrases -- strings to be compared to candidate names in case ID is not available

    """
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

def obtain_edge_features(phrase, dag, k, j):
    """Construct feature vector for edge label detection; Unused

    Keyword arguments:
    phrase -- list of phrases to determine variable phrase
    dag -- DAG of the question
    k -- edge target index
    j -- edge start index

    """
    q_type = ['who ', 'what ', 'when ', 'where ', 'how many ', 'where ','which ', 'how ', 'whom ']
    q_var = [V[0] for V in phrase if V[1] == 3]
    f6_11 = [0] * len(q_type)
    if q_var[0] in q_type and len(q_var) > 0:
        f6_11[q_type.index(q_var[0])] = 1
    ents = len([P for P in phrase if P[1] == 0])
    f1 = len(dag[j])
    f2 = ents
    f3 = phrase[j][1]
    f4 = phrase[k][1]
    f5 = len([item for sublist in dag for item in sublist])
    return [f1, f2, f3, f4, f5] + f6_11

def get_edge_features(dct, phrase, k, j):
    """Construct feature vector for edge label detection

    Keyword arguments:
    dct -- complete vocabulary of the corpus; pairs (word, id)
    phrase -- list of phrases to determine variable phrase
    k -- edge target index
    j -- edge start index

    """
    feature = [0] * (2 * len(dct.keys()) + 8)

    # start
    f3 = phrase[j]
    start_phr = f3[0].strip().split()
    start_tag = f3[1]

    # target
    f4 = phrase[k]
    trg_phr = f4[0].strip().split()
    trg_tag = f4[1]

    feature[start_tag] = 1
    feature[trg_tag + 4] = 1
    for phr in start_phr:
        feature[dct[phr]] = 1
    for phr in trg_phr:
        feature[dct[phr] + 8 + len(dct.keys())] = 1
    return feature

def edge_gold_standard(phrases, dags, gold_edges):
    """Construct features and determine correct labels for all edges

    Keyword arguments:
    phrases -- list of lists of phrases
    dags -- list of DAGs for all questions
    gold_edges -- correct labels for edges

    """
    result = []
    features = []
    dct = pickle.load(open("edge_dict.pickle"))
    bow_dct = pickle.load(open("bow_all_words_dict.pickle"))
    for i in range(len(dags)):
        phrase = phrases[i]
        dag = dags[i]
        gold = gold_edges[i]
        if len(dag) == len(phrase):
            for j in range(len(dag)):
                d = dag[j]
                g = gold[j]
                if len(d) > 0:
                    for k in d:
                        features.append(get_edge_features(bow_dct, phrase, k, j))
                    result += [dct[e_label] for e_label in g]
    return result, features

def lemmatize_word(text):
    """Lemmatize text using nltk library

    Keyword arguments:
    text -- list of strings to be lemmatized

    """
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(t,'v') for t in text]

def label_edges(phrase, dag, w, j):
    """Label all edges going out of a node given a trained model

    Keyword arguments:
    phrase -- list of phrases
    dag -- DAG of question
    w -- chosen model
    j -- index of node

    """
    d = pickle.load(open("C:\\Users\\Martin\\PycharmProjects\\xserpy\\query_intention\\edge_dict.pickle"))
    dct = [''] * len(d.keys())
    for D in d.keys():
        dct[d[D]] = D
    result = ['x'] * len(phrase)
    for d in dag[j]:
        feature = obtain_edge_features(phrase, dag, d, j)
        label = w.predict(feature)
        result[d] = dct[label[0]]
    return result

def label_entity(phrase, w, candidate):
    """Link a phrase to entity given a trained model

    Keyword arguments:
    phrase -- entity phrase
    w -- chosen model
    candidate -- list of candidates for the entity

    """
    feature = get_feature(phrase, candidate)
    label = w.predict(feature)[0]
    if label < len(candidate):
        return candidate[label]['mid'].replace('/','.')[1:]
    elif len(candidate) > 0:
        return candidate[0]['mid'].replace('/','.')[1:]
    else:
        return 'x'

def label_relation(phrase, w, bow_dct, g_dct):
    """Link a phrase to relation given a trained model

    Keyword arguments:
    phrase -- relation phrase
    w -- chosen model
    bow_dct -- vocabulary of all words in relation phrases in dataset
    g_dct -- dictionary of all correct relation labels in dataset

    """
    r_bow, v_bow = get_idx(phrase, bow_dct)
    features = construct_relation_features([r_bow],[v_bow], len(bow_dct))[0]
    label = w.predict(features)[0]
    for k in g_dct.keys():
        if label == g_dct[k]:
            return k
    return 'relation'

def label_all(phrase, dag, candidates, ent_path, ed_path, rel_path, bow_path, g_path):
    """Link all phrases and edges in on question to knowledge base given a trained model

    Keyword arguments:
    phrase -- list of phrases in the question
    dag -- DAG of the question
    candidates -- list of candidates for the entities
    ent_path -- path to entity model
    ed_path -- path to edge model
    rel_path -- path to relation model
    bow_path -- path to dictionary
    g_path -- path to dictionary

    """
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
            if len(candidates) > 0:
                result.append(label_entity(phrase[i], ent_perc, candidates[e]))
                e += 1
            else:
                result.append('x')
        elif phrase[i][1] == 1:
            result.append(label_relation(phrase, rel_lr, bow_dct, g_dct))
        if dag[i]:
            result += label_edges(phrase, dag, edge_perc, i)
        else:
            result += ['x'] * (len(phrase))
    return result

def get_bow(phrases, rel):
    """Construct a bag-of-words model over relation phrases

    Keyword arguments:
    phrase -- list of lists of phrases
    ent -- list of correct relation labels

    """
    q_type = ['who ', 'what ', 'when ', 'where ', 'how many ', 'where ']
    v_bow = []
    bow_dct = pickle.load(open("bow_dict_all.pickle"))
    g_dct = pickle.load(open("rel_dict.pickle"))
    gold = []
    r_bow = []
    i = 0
    k = 0
    for j in range(len(phrases)):
        phrase = phrases[j]
        g = rel[j]
        # g = [pp for pp in ents if pp[:3] != 'en.' and pp[:2] != 'm.']
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
                # if R not in bow_dct.keys():
                #     bow_dct[R] = i
                #     i += 1
                rr.append(bow_dct[R])
            r_bow.append(rr)
        for G in g:
            # if G not in g_dct.keys():
            #     g_dct[G] = k
            #     k += 1
            gold.append(g_dct[G])
    return bow_dct, v_bow, r_bow, gold

def construct_relation_features(r_bow, v_bow, length):
    """Construct features for relation linking

    Keyword arguments:
    r_bow -- list of lists of indexes of relations in questions
    v_bow -- list of lists of indexes of variables in questions
    length -- feature vector size

    """
    features = []
    for i in range(len(r_bow)):
        feature = [0] * (length + 7)
        r = r_bow[i]
        v = v_bow[i]
        if v >= 0:
            feature[v] = 1
        for R in r:
            feature[R + 7] = 1
        features.append(feature)
    return features

def get_idx(phrase, bow_dct):
    """Obtain indexes of relation and variable words in a question

    phrase -- a question
    bow_dct -- vocabulary of relation words

    """
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

def parse_log_formula(q, cvt):
    """Read logic formula from the dataset and construct a gold standard for linking and DAGs

    Keyword arguments:
    q -- question
    cvt -- string to be used as secondary variable

    """
    gold = []
    entities = []
    relations = []
    dag = []
    edges = []
    formula = q.targetFormula
    if formula[:6] == '(count':
        formula = formula[7:-1]
    formula = formula[1:-1]
    i = 0
    while formula[i] != ' ':
        i += 1
    variable = formula[:i]
    condition = formula[i+1:]

    if len(variable.split()) == 1 and len(condition.split()) == 1:
        first = 'PO'
        second = 'SP'
        relations = [variable[4:]]
        if variable[0] != '!':
            relations = [variable[3:]]
            first = 'SP'
            second = 'PO'
        # gold = ['?x', 'x', 'x', 'x', variable[4:], first, 'x', second, condition[3:], 'x', 'x', 'x']
        entities = [condition[3:]]
        dag = [[], [0, 2], []]
        edges = [[], [first, second], []]

    elif condition[:8] == '((lambda':
        cond = condition.split()
        ent =  cond[-1][3:-1]
        if cond[2][1] == '!':
            edge = cond[2][5:]
            # gold = ['?x', 'x', 'x', 'x', cvt, variable[4:], 'x', 'x', ent, 'x', edge, 'x']
            entities = [ent]
            relations = [cvt]
            dag = [[], [0], [1]]
            edges = [[], [variable[4:]], [edge]]

        else:
            edge = cond[2][4:]
            # gold = ['?x', 'x', 'x', 'x', cvt, variable[4:], 'x', edge, ent, 'x', 'x', 'x']
            entities = [ent]
            relations = [cvt]
            dag = [[], [2], [0]]
            edges = [[], [variable[4:]], [edge]]

    elif condition[:4] == '(and':
        condition = condition[5:-1]
        if condition[:8] == '((lambda':
            cond = condition.split('((lambda x ')[1:]
            # gold = ['?x', 'x', 'x'] + (['x'] * len(cond)) + [cvt, variable[4:], 'x']
            # ents = []
            entities = []
            edges = [[], [variable[4:]]]
            for c in cond:
                C = c.split()
                if not 'date' in c:
                    # gold += [C[0][4:]]
                    # ents += [C[-1][3:-1]] + ['x'] * (2 + len(cond))
                    entities += [C[-1][3:-1]]
                else:
                    # gold += [C[0][4:]]
                    if '-1' in c:
                        # ents += ['\"' + C[-3] + '\"^^xsd:gYear'] + ['x'] * (2 + len(cond))
                        entities += ['\"' + C[-3] + '\"^^xsd:gYear']
                    else:
                        return []
                if C[0][1] == '!':
                    edges[1] += [C[0][5:]]
                else:
                    edges[1] += [C[0][4:]]
            # gold += ents
            relations = [cvt]
            dag = [[]] + [[0] + range(2, len(cond) + 2)] + [[]] * len(cond)
            edges += [[]] * len(cond)
        else:
            condition = condition[5:]
            cond = condition.split('((lambda x ')[1:]
            for i in range(1, len(cond)-1):
                cond[i] = cond[i][:-2]
            # gold = ['?x', 'x', 'x'] + (['x'] * len(cond)) + [cvt, variable[4:], 'x']
            # ents = []
            entities = []
            edges = [[], [variable[4:]]]
            for c in cond:
                C = c.split()
                edges[1] += [C[0][4:]]
                if not 'date' in c:
                    # gold += [C[0][4:]]
                    # ents += [C[-1][3:-1]] + ['x'] * (2 + len(cond))
                    entities += [C[-1][3:-1]]
                else:
                    # gold += [C[0][4:]]
                    if '-1' in c:
                        # ents += ['\"' + C[-3] + '\"^^xsd:gYear'] + ['x'] * (2 + len(cond))
                        entities += ['\"' + C[-3] + '\"^^xsd:gYear']
                    else:
                        return [], [], [], []
            # gold += ents
            relations = [cvt]
            dag = [[]] + [[0] + range(2, len(cond) + 2)] + [[]] * len(cond)
            edges += [[]] * len(cond)
    else:
        cond = condition.split()
        if cond[0][1] == '!':
            edge = cond[0][5:]
            ent = cond[1][3:-1]
            # gold = ['?x', 'x', 'x', 'x', cvt, variable[4:], 'x', 'x', ent, 'x', edge, 'x']
            entities = [ent]
            relations = [cvt]
            dag = [[], [0], [1]]
            edges += [[], [variable[4:]], [edge]]
        else:
            if cond[0][:3] == 'fb:':
                # gold = [cond[2][3:-1], 'x', 'x', 'x', cond[1][4:], 'PO', 'x', 'SP', cond[0][3:], 'x', 'x', 'x']
                entities = [cond[0][3:]]
                relations = [cond[1][4:]]
                dag = [[], [0, 2], []]
                edges += [[], ['PO', 'SP'], []]
            elif cond[0] == '(number':
                # gold = ['?x', 'x', 'x', 'x', variable[3:], 'SP', 'x', 'PO', cond[1][:-2], 'x', 'x', 'x']
                entities = [cond[1][:-2]]
                relations = [variable[3:]]
                dag = [[], [0, 2], []]
                edges += [[], ['SP', 'PO'], []]
    return entities, relations, edges, dag

def create_dicts(gold_edges, rels):
    """Create dictionaries for all relation and edge labels

    Keyword arguments:
    gold_edges -- labels of edges
    rels -- labels of relations

    """
    ed = list(enumerate(list(set([item for sublist in gold_edges for subsublist in sublist for item in subsublist]))))
    ed_dct = dict([(item, index) for (index, item) in ed])
    rel = list(enumerate(list(set([item for sublist in rels for item in sublist]))))
    rel_dct = dict([(item, index) for (index, item) in rel])
    return ed_dct, rel_dct

def check_features(X, y):
    """Check if features were correctly created

    Keyword arguments:
    X -- feature vectors
    y -- labels

    """
    bow_dct = pickle.load(open("bow_dict_all.pickle"))
    g_dct = pickle.load(open("rel_dict.pickle"))
    for x in range(len(X)):
        xx = X[x]
        xxx = [i for i in range(len(xx[7:])) if xx[i+7] == 1]
        words = [b for b in bow_dct.keys() if bow_dct[b] in xxx]
        relation = [g for g in g_dct.keys() if g_dct[g] == y[x]]
        a = []

def evaluate_model(X, gold, dct, csf, phrases):
    """Compute error of the model

    Keyword arguments:
    X -- feature vectors
    gold -- labels
    dct -- vocabulary of relations
    csf -- trained model
    phrases -- questions

    """
    correct = 1.0
    rev_dict = dict([(dct[key],key) for key in dct.keys()])
    for i in range(len(X)):
        label = csf.predict(X[i])[0]
        rel = rev_dict[label]
        if gold[i][0] == rel or (gold[i][0][0] == '?' and rel[0] == '?'):
            correct += 1.0
        else:
            print phrases[i][1], rel, gold[i][0]
    print correct/len(X)

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

    # Mode for obtaining candidates from Freebase
    if 'e' in type:
        candidates = obtain_entity_candidates(phrases, n_cand)
        pickle.dump(candidates, open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle","wb"))

    # Mode for obtaining gold standard of entities
    if 'g' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        ent = pickle.load(open('query_gold_ent_' + mode + '.pickle'))
        labels = obtain_entity_labels(candidates, ent, phrases)
        pickle.dump(labels, open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle","wb"))

    # Mode for obtaining features of entities
    if 'f' in type:
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + GOLD + "_" + mode + "_" + str(size) + ".pickle"))
        features = get_features(phrases,candidates)
        F = [features[i] for i in range(len(features)) if labels[i] >= 0]
        L = [label for label in labels if label >= 0]
        pickle.dump(F,open(path + "data" + sep + FEATURES + "_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(L,open(path + "data" + sep + LABELS + "_" + mode + "_" + str(size) + ".pickle","wb"))

    # Mode for training model for linking entities
    if 't' in type:
        perc = OneVsRestClassifier(LogisticRegression(verbose=0, max_iter=100))
        y = pickle.load(open(path + "data" + sep + LABELS + "_trn_641.pickle"))
        X = pickle.load(open(path + "data" + sep + FEATURES + "_trn_641.pickle"))
        w = perc.fit(X,y)
        pickle.dump(w, open("ent_lr_trn_641.pickle","wb"))
        y_tst = pickle.load(open(path + "data" + sep + LABELS + "_tst_276.pickle"))
        X_tst = pickle.load(open(path + "data" + sep + FEATURES + "_tst_276.pickle"))
        print perc.score(X_tst, y_tst)

    # Mode for training model for linking relations
    if 'r' in type:
        ent = pickle.load(open('query_gold_rel_trn.pickle'))
        bow_dct, v_bow, r_bow, y = get_bow(phrases, ent)
        length = len(bow_dct.keys())
        X = construct_relation_features(r_bow, v_bow, length)
        csf = LinearSVC(max_iter=10000)
        csf.fit(X, y)
        pickle.dump(csf, open("relation_lr_trn_641.pickle","wb"))

    # Mode for obtaining gold standard and features of edges
    if 'l' in type:
        dags = pickle.load(open("gold_dags_" + mode + "_" + str(size) + ".pickle"))
        gold_edges = pickle.load(open("query_gold_edges_" + mode + ".pickle"))
        result, features = edge_gold_standard(phrases, dags, gold_edges)
        pickle.dump(result, open(path + "query_intention" + sep + "edge_labels_" + mode + ".pickle","wb"))
        pickle.dump(features, open(path + "query_intention" + sep + "edge_features_" + mode + ".pickle", "wb"))

    # Mode for training and evaluating model for linking edges
    if 'd' in type:
        perc = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=100))
        y = pickle.load(open(path + "query_intention" + sep + "edge_labels_trn.pickle"))
        X = pickle.load(open(path + "query_intention" + sep + "edge_features_trn.pickle"))
        w = perc.fit(X,y)
        pickle.dump(w, open("edge_lr_trn.pickle","wb"))
        y_tst = pickle.load(open(path + "query_intention" + sep + "edge_labels_tst.pickle"))
        X_tst = pickle.load(open(path + "query_intention" + sep + "edge_features_tst.pickle"))
        print perc.score(X_tst,y_tst)

    # Testing mode for linking whole questions
    if 'a' in type:
        dags = parse_dags(phrases)
        candidates = pickle.load(open(path + "data" + sep + CANDIDATES + "_" + mode + "_" + str(size) + ".pickle"))
        mapped = []
        for i in range(len(phrases)):
            mapped.append(label_all(phrases[i], dags[i], candidates[i], "ent_lr_trn_641.pickle", "edge_lr_trn.pickle", "relation_lr_trn_641.pickle" "bow_dict_tst_276.pickle", "rel_dict.pickle"))
        pickle.dump(mapped, open("emapped.pickle","wb"))

    # Mode for parsing gold standards from logic formulas
    if 'q' in type:
        alphabets = [chr(c) for c in range(97,106)]
        keywords = ['?' + ''.join(i) for i in product(alphabets, repeat = 3)]
        entities = []
        relations = []
        edges = []
        dags = []
        for q in range(len(questions)):
            ent, rel, edg, dag = parse_log_formula(questions[q], keywords[q])
            entities.append(ent)
            dags.append(dag)
            edges.append(edg)
            relations.append(rel)
        pickle.dump(entities, open('query_gold_ent_' + mode + '.pickle','wb'))
        pickle.dump(relations, open('query_gold_rel_' + mode + '.pickle','wb'))
        pickle.dump(dags, open('query_gold_dags_' + mode + '.pickle','wb'))
        pickle.dump(edges, open('query_gold_edges_' + mode + '.pickle','wb'))

    # Mode for creating dictionaries
    if 'c' in type:
        gold_edges = pickle.load(open("query_gold_edges_trn.pickle")) + pickle.load(open("query_gold_edges_tst.pickle"))
        rel = pickle.load(open('query_gold_rel_trn.pickle')) + pickle.load(open('query_gold_rel_tst.pickle'))
        ed_dct, rel_dct = create_dicts(gold_edges, rel)
        pickle.dump(rel_dct,open("rel_dict.pickle","wb"))
        pickle.dump(ed_dct,open("edge_dict.pickle","wb"))