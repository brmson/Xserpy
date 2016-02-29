import json,pickle,random
from annotate.annotator import object_decoder,parse_phrases,parse_dags
from phrase_detector import train,compute_score

class Instance:
    def __init__(self,sentence,candidates,dependencies,label):
        self.sentence = sentence
        self.candidates = candidates
        self.dependencies = dependencies
        self.label = label

def score(buff,weights,cl):
    result = []
    for b in buff:
        sc = compute_score(b,weights,0)
        result.append(sc)
    res = zip(buff,result)
    res.sort(key=lambda x: x[1], reverse=True)
    result = [r[0] for r in res]
    return result

def beam_search(instance,size,phrase_labels,dependency_labels,weights):
    beam = [[]]
    for i in range(len(instance.sentence)):
        buff = []

        for z in beam:
            for p in phrase_labels:
                buff.append(z + p)

        beam = score(buff,weights,len(phrase_labels))[:size]
        index = i*(len(instance.sentence)+1)
        label = instance.sentence_label[index]
        if label not in beam:
            return beam[0]

        for c in range(len(instance.candidates)):
            buff = []
            for z in beam:
                buff.append(z + dependency_labels[0])
                if c in instance.dependencies[index]:
                    for d in dependency_labels:
                        buff.append(z + d)

            beam = score(buff,weights,len(dependency_labels))[:size]
            label = instance.label[index+1:index+len(instance.sentence)+1]
            if label not in beam:
                return beam[0]

    return beam[0]

def train_with_beam(n_iter,examples,weights,size,phrase_labels,dependency_labels):
    learning_rate = 1
    for i in range(n_iter):
        err = 0
        for features, true in examples:
            guess = beam_search(features,size,phrase_labels,dependency_labels,weights)
            if guess != true:
                for f in features:
                    weights[f][true] += learning_rate
                    weights[f][guess] -= learning_rate
                err += 1.0
        random.shuffle(examples)
    print err/len(examples)
    return weights

def get_db_entities(questions):
    types = []
    result = []
    for q in questions:
        letter = [0,0]
        j = 0
        i = 0
        strings = []
        tF = q.targetFormula
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
        types.append(temp)
        result.append(strings)
    types = list(set(types))
    return result

def get_entities_relations(entities):
    all = [item for sublist in entities for item in sublist]
    entities = []
    relations = []
    for a in all:
        if a[:3] == 'en.':
            entities.append(a)
        else:
            relations.append(a)
    return list(set(relations)),list(set(entities))

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
    # pickle.dump(result,open("entities_100.pickle","w"))
    return result

def gold_standard(phrases,dags,entities):
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
                     if inp == 'x':
                         res.append(inp)
                     else:
                         res.append(entity[int(inp)])
            k += len(dag)+1
        print res
        # result.append(res)
    return result

def create_features(questions,phrases):
    phr = list(set([p[0].lower().strip() for sublist in phrases for p in sublist]))
    rel_entities = get_db_entities(questions)
    relations,entities = get_entities_relations(rel_entities)
    surf_names = get_surface_names(entities)
    e_features = {}
    for entity in entities:
        feature = construct_feature(phr,entity,surf_names)
        # e_features.append(feature)
        e_features[entity] = feature
    return e_features

def construct_feature(phr,entity,surf_names):
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
        overlap = len(s) - len(s.replace(ph,''))
        if overlap > 0:
            feature.append("over_"+name+"_"+ph+"_"+str(overlap))
    return feature

def create_examples(questions,phrases,dags):
    gold = pickle.load(open('query_gold_10.pickle'))
    e_features = create_features(questions,phrases)
    instances = []
    for i in range(10):
        instances.append(Instance(phrases[i],e_features,dags[i],gold[i]))
    return instances

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\data\\free917.train.examples.canonicalized.json"
    questions = json.load(open(path),object_hook=object_decoder)[:10]
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    labels = pickle.load(open(path+"data\\questions_trn_100.pickle"))
    phrases = parse_phrases(questions,labels)
    dags = parse_dags(phrases)
    # features = create_features(questions,phrases)
    rel_entities = get_db_entities(questions)
    relations,entities = get_entities_relations(rel_entities)
    # surf_names = get_surface_names(entities)
    gold = gold_standard(phrases,dags,rel_entities)
    pickle.dump(gold,open('query_gold_10.pickle','wb'))