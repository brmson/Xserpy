import json
import pickle
import random
import argparse
import os

from annotate.annotator import object_decoder, parse_phrases, parse_dags
from phrase_detection.phrase_detector import compute_score


class Instance:
    def __init__(self, sentence, candidates, dependencies, label):
        self.sentence = sentence
        self.candidates = candidates
        self.dependencies = dependencies
        self.label = label

def score(buff, weights, cl):
    result = []
    for b in buff:
        sc = compute_score(b, weights, 0)
        result.append(sc)
    res = zip(buff, result)
    res.sort(key=lambda x: x[1],  reverse=True)
    result = [r[0] for r in res]
    return result

def beam_search(instance, size, dependency_labels, weights, training=True):
    beam = [[]]
    for i in range(len(instance.sentence)):
        buff = []

        for z in beam:
            for p in instance.candidates:
                buff.append(z + [p])
        random.shuffle(buff)
        beam = score(buff, weights, len(instance.candidates))[:size]
        index = i*(len(instance.sentence)+1)
        label = instance.label[index]
        if label not in [b[-1] for b in beam] and training:
            return beam[0]

        for c in range(len(instance.sentence)):
            buff = []
            for z in beam:
                buff.append(z + [dependency_labels[0]])
                if c in instance.dependencies[i]:
                    for d in dependency_labels:
                        buff.append(z + [d])
                # else:
                #     buff.append(z + ['x'])
            random.shuffle(buff)
            beam = score(buff, weights, len(dependency_labels))[:size]
            label = instance.label[index+1:index+c+3]
            if label not in beam  and training:
                return beam[0]

    return beam[0]

def train_with_beam(n_iter, examples, weights, size, dependency_labels, features):
    learning_rate = 1
    for i in range(n_iter):
        err = 0
        for instance in examples:
            true = instance.label
            guess = beam_search(instance, size, ['x']+dependency_labels, weights)

            if guess != true:
                for j in range(len(guess)):
                    if guess[j] != true[j]:
                        for f in features[guess[j]]:
                            weights[f][true[j]] += learning_rate
                            weights[f][guess[j]] -= learning_rate
                err += 1.0

        random.shuffle(examples)
        print err/len(examples)
    return weights

def get_entity(tF):
    letter = [0, 0]
    j = 0
    i = 0
    strings = []
    temp = ""
    while i < len(tF):
        if tF[i] == 'f':
            strin = ""
            i += 3
            while tF[i] != ')' and tF[i] != ' ':
                strin += tF[i]
                i += 1
            strings.append(strin)
            if strin[:2] == 'en':
                temp += 'E'
                j = 0
            else:
                temp += 'C'
                j = 1
            temp += str(letter[j])
            letter[j] += 1
        else:
            if tF[i:i+3] == 'dat':
                temp += 'D'
                while tF[i] != ')':
                    i += 1
            temp += tF[i]
            i += 1
    return strings, temp

def get_db_entities(questions):
    simple = []
    result = []
    for q in questions:
        strings,  temp = get_entity(q.targetFormula)
        result.append(strings)
        if len(temp) == 8:
            simple.append(q)
    return (result, simple)

def get_entities_relations(entities):
    all = [item for sublist in entities for item in sublist]
    entities = []
    relations = ['SC','SP','PO']
    for a in all:
        if a[:3] == 'en.' or a[:2] == 'm.':
            entities.append(a)
        else:
            relations.append(a)
    return list(set(relations)), list(set(entities))

def get_surface_names(entities):
    result = {}
    for e in entities:
        surf = ""
        s = e.split('.')
        w = s[1].split('_')
        for W in w:
            if W == 'the':
                continue
            surf += W + " "
        result[e] = surf[:-1]
    # pickle.dump(result, open("entities_100.pickle", "w"))
    return result

def gold_standard(phrases, dags, entities):
    result = []
    for i in range(len(entities)):
        phrase = phrases[i]
        entity = entities[i]
        dag = dags[i]
        temp = []
        for e in entity:
            print e
        for j in range(len(dag)):
            temp.append(phrase[j][0])
            for d in range(len(dag)):
                if d in dag[j]:
                    temp.append(phrase[int(d)][0])
                else:
                    temp.append('x')
        k = 0
        res = []
        while k < len(temp):
            T = temp[k:k+len(dag)+1]
            print T
            for t in T:
                 if t == 'x':
                     res.append(t)
                 else:
                     inp = raw_input(t+" ")
                     if inp == 'x' or inp == 'SC' or inp == 'SP' or inp == 'PO':
                         res.append(inp)
                     else:
                         res.append(entity[int(inp)])
            k += len(dag)+1
        # print res
        result.append(res)
    return result

def create_features(questions, phrases):
    scores = pickle.load(open("scores_trn_100.pickle"))
    phr = list(set([p[0].lower().strip() for sublist in phrases for p in sublist]))
    rel_entities, simple = get_db_entities(questions)
    relations, entities = get_entities_relations(rel_entities)
    surf_names = get_surface_names(entities)
    e_features = {}
    r_features = {}
    for entity in entities:
        feature = construct_feature(phr, entity, surf_names,scores)
        # e_features.append(feature)
        e_features[entity] = feature
    e_features['x'] = []
    for relation in relations:
        e_features[relation] = []
    # r_features['x'] = []
    return e_features # + r_features

def construct_feature(phr, entity, surf_names, scores):
    feature = []
    name = entity.split('.')[-1]
    for ph in phr:
        s = surf_names[entity]
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
        if entity in scores.keys():
            feature.append('pop_score_'+str(scores[entity]))
    return feature

def create_examples(questions, phrases, dags, goldname):
    gold = pickle.load(open(goldname))
    e_features = create_features(questions, phrases)
    instances = []
    for i in range(len(gold)):
        instances.append(Instance(phrases[i], generate_candidates(questions[i], e_features), dags[i], gold[i]))
    return instances, e_features

def init_weights(features, weights):
    for k in features.keys():

        for f in features[k]:
            if f not in weights.keys():
                weights[f] = {}
                for j in features.keys():
                    weights[f][j] = 0
                weights[f]['x'] = 0

    return weights

def generate_candidates(question, features):
    strings, temp = get_entity(question.targetFormula)
    candidates = features.keys()[:]
    for s in strings:
        candidates.remove(s)
    if len(candidates) >= len(strings):
        indices = random.sample(xrange(len(candidates)), len(strings))
        for i in indices:
            strings.append(candidates[i])
    return strings

if __name__ == "__main__":
    sep = os.path.sep

    parser = argparse.ArgumentParser(description="Train weights for detecting query intention or create gold standard")
    parser.add_argument("fpath", help="filepath", type=str)
    parser.add_argument("start", help="start", type=int, default=0)
    parser.add_argument("end", help="end", type=int, default=0)
    parser.add_argument("n_iter", help="Number of iterations", type=int, default=0)
    parser.add_argument("size", help="Beam size", type=int, default=0)
    parser.add_argument("type", help="type", type=str)
    parser.add_argument("gold", help="goldname", type=str)
    args = parser.parse_args()

    path = args.fpath
    start = args.start
    end = args.end
    goldname = args.gold

    questions = json.load(open(path+"data" + sep + "free917.train.examples.canonicalized.json"), object_hook=object_decoder)[:100]
    labels = pickle.load(open(path +"data" + sep + "questions_trn_100.pickle"))

    phrases = parse_phrases(questions, labels)
    # candidates = obtain_pop_score(phrases) # pickle.load(open("candidates_trn_100.pickle"))#
    # pickle.dump(candidates,open("scores_trn_100.pickle","wb"))

    if 't' in args.type:
        phrases = parse_phrases(questions, labels)
        dags = parse_dags(phrases)
        rel_entities, simple = get_db_entities(questions)
        relations, entities = get_entities_relations(rel_entities)
        examples, features = create_examples(questions, phrases, dags, goldname)
        # features = get_features(phrases, candidates)
        # examples = obtain_examples(phrases, candidates, dags, goldname)
        W = train_with_beam(args.n_iter, examples, init_weights(features, {}), args.size, ['SP','SC','PO','x'], features)
        #pickle.dump(W, open(path+"models" + sep + "w_qint_" + str(args.size) + "_i" + str(args.n_iter) + "_ffs.pickle", "wb"))
    else:
        questions = questions[start:end]
        labels = labels[start:end]
        phrases = parse_phrases(questions, labels)
        dags = parse_dags(phrases)
        rel_entities, simple = get_db_entities(questions)
        gold = gold_standard(phrases, dags, rel_entities)
        pickle.dump(gold, open('query_gold_'+str(start)+'_'+str(end)+'.pickle', 'wb'))