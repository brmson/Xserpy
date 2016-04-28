import pickle, os, argparse

from nltk.stem.wordnet import WordNetLemmatizer

from query_intention.fb_query import *
from query_instantiate import Instance,parse_to_phrases,object_decoder


def obtain_feature(phrase, candidates):
    f = {}
    for c in candidates:
        feature = []
        ph = phrase[0]
        s = c['name']
        name = s.replace(' ','_')
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
        feature.append('pop_'+str(c['score']))
        f[c['mid']] = feature
    return f

def get_features(phrases, candidates):
    features = {}
    for i in range(len(phrases)):
        for j in range(len(phrases[i])):
            features.update(obtain_feature(phrases[i][j],candidates[i][j]))
    features['x'] = []
    return features

def obtain_examples(phrases, candidates, dags, goldname):
    gold = pickle.load(open(goldname))
    instances = []
    for i in range(len(gold)):
        instances.append(Instance(phrases[i], candidates[i], dags[i], gold[i]))
    return instances


def obtain_candidates(phrases):
    size = 5
    result = []

    for sentence in phrases:
        sent = []
        for phrase in sentence:
            if phrase[1] == 0:
                cand = query_freebase_entity(phrase[0], 'freebase', size)
                sent.append(cand)
        result.append(sent)
    return result

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
    size = args.size

    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    labels = pickle.load(open(path +"data" + sep + "questions_" + mode + "_" + str(size) + ".pickle"))

    if 'e' in type:
        phrases = parse_to_phrases(questions, labels)
        candidates = obtain_candidates(phrases, n_cand)
        pickle.dump(candidates,open(path + "data" + sep + "candidates_" + mode + "_" + str(size) + ".pickle","wb"))
    if 'r' in type:
        pickle.load(open(path + "data" + sep + "candidates_" + mode + "_" + str(size) + ".pickle"))