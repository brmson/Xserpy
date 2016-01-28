from annotate.annotator import Question,object_decoder,json
import pickle

class Phrase(object):
    def __init__(self,variable,entity,relation,category):
        self.variable = variable
        self.entity = entity
        self.relation = relation
        self.category = category

def parse_to_phrases(questions,labels):
    index = 0
    phrases = []
    for q in questions:
        u = q.utterance.split()
        dic = {}
        phrase = ["","","",""]
        order = [0,0,0,0]
        j = 0
        for word in u:
            l = labels[index]
            index += 1
            if l == 4:
                continue
            if l not in dic.keys():
                dic[l] = j
                order[j] = l
                j += 1
            phrase[dic[l]] += word + " "
        phrases.append(zip(phrase,order))
    return phrases

if __name__ == "__main__":

    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    questions = json.load(open(path+"data\\free917.train.examples.canonicalized.json"),object_hook=object_decoder)
    labels = pickle.load(open(path+"data\\labels_trn_90.pickle"))
    phrases = parse_to_phrases(questions[:20],labels)
    print ""
