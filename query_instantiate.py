
class Instance:
    def __init__(self,sentence,candidates,dependencies,label,dependency_label):
        self.sentence = sentence
        self.candidates = candidates
        self.dependencies = dependencies
        self.sentence_label = label
        self.dependency_label = dependency_label

def label_phrase(phrase):
    return phrase

def instantiate_query_intention(instance,size,phrase_labels,dependency_labels):
    beam = [[]]
    for i in range(len(instance.sentence)):
        buff = []

        x = instance.sentence[i]

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
                    # buff.append((i,c))
                    for d in dependency_labels:
                        buff.append(z + d)

            beam = buff[:size]
            label = instance.dependency_label[c][:i]
            if label not in beam:
                return beam[0]

    return beam[0]
