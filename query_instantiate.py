import json,pickle
from annotate.annotator import object_decoder,parse_phrases,parse_dags
class Instance:
    def __init__(self,sentence,candidates,dependencies,label,dependency_label):
        self.sentence = sentence
        self.candidates = candidates
        self.dependencies = dependencies
        self.sentence_label = label
        self.dependency_label = dependency_label


def beam_search(instance,size,phrase_labels,dependency_labels):
    beam = [[]]
    for i in range(len(instance.sentence)):
        buff = []

        for z in beam:
            for p in phrase_labels:
                buff.append(z + p)
        beam = buff[:size]

        label = instance.sentence_label[:i]
        if label not in beam:
            return beam[0]

        for c in range(len(instance.candidates)):
            buff = []
            for z in beam:
                buff.append(z + dependency_labels[0])
                if (i,c) in instance.dependencies.keys():
                    for d in dependency_labels:
                        buff.append(z + d)

            beam = buff[:size]
            label = instance.dependency_label[c][:i]
            if label not in beam:
                return beam[0]

    return beam[0]

def get_db_entities(questions):
    result = []
    for q in questions:
        i = 0
        strings = []
        tF = q.targetFormula
        while i < len(tF):
            if tF[i] == 'f':
                str = ""
                i += 3
                while tF[i] != ')' and tF[i] != ' ':
                    str += tF[i]
                    i += 1
                strings.append(str)
            else:
                i += 1
        result.append(strings)
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
    pickle.dump(result,open("entities_100.pickle","w"))

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
if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\data\\free917.train.examples.canonicalized.json"
    questions = json.load(open(path),object_hook=object_decoder)[:100]
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    # labels = pickle.load(open(path+"data\\questions_trn_100.pickle"))
    # phrases = parse_phrases(questions,labels)
    # dags = parse_dags(phrases)
    rel_entities = get_db_entities(questions)
    relations,entities = get_entities_relations(rel_entities)
    get_surface_names(entities)
    # gold_standard(phrases,dags,entities)
