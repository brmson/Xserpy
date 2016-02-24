import json
from annotate.annotator import object_decoder
class Instance:
    def __init__(self,sentence,candidates,dependencies,label,dependency_label):
        self.sentence = sentence
        self.candidates = candidates
        self.dependencies = dependencies
        self.sentence_label = label
        self.dependency_label = dependency_label


def instantiate_query_intention(instance,size,phrase_labels,dependency_labels):
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

def gold_standard(questions):
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
        print strings

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\data\\free917.train.examples.canonicalized.json"
    questions = json.load(open(path),object_hook=object_decoder)
    gold_standard(questions)
