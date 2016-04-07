from fb_query import *
from query_instantiate import Instance, beam_search, train_with_beam, score
import pickle,nltk
from nltk.stem.wordnet import WordNetLemmatizer

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
    scoring = ['freebase','schema','schema']
    category = [False, True, True]
    result = []

    for sentence in phrases:
        sent = []
        for phrase in sentence:
            cand = []
            if phrase[1] != 3:
                i = phrase[1]
                cand = query_freebase(phrase[0], scoring[i], size, category[i])
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
            f = query_freebase(e[0], scoring[i], 10, id, category[i])
            if f is None:
                scores['en.' + id] = 0
            else:
                scores['en.' + id] = f['score']
    return scores

def lemmatize_word(text):
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(t,'v') for t in text]

if __name__ == "__main__":
    print lemmatize_word(['did','died'])