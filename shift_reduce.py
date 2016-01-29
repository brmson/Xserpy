from annotate.annotator import Question,object_decoder,json
import pickle

class Item(object):
    def __init__(self,stack,queue):
        self.stack = stack
        self.queue = queue

def compute_score(item):
    return 1

def shift(item):
    q = item.queue[0]
    s = item.stack
    s.append(q)
    return Item(s,item.queue[1:])

def reduce_item(item):
    s = item.stack
    s.pop()
    return Item(s,item.queue)

def arcleft(item):
    return

def arcright(item):
    return

def shift_reduce(sentence):
    actions = [shift,reduce_item,arcleft,arcright]
    deque = []
    deque.append(Item([],sentence,False))
    result = None
    score = 0
    while deque:
        lst = []
        for item in deque:
            for action in actions:
                new_item = action(item)
                if not new_item.queue:
                    new_score = compute_score(new_item)
                    if result == None or new_score > score:
                        result = new_item
                        score = new_score
                else:
                    lst.append(new_item)
        deque = lst[:10]
    return result

def parse_to_phrases(questions,labels):
    phrases = []
    print len(questions),len(labels)
    for i in range(len(questions)):
        u = questions[i].utterance.split()
        label = labels[i]
        dic = {}
        phrase = ["","","",""]
        order = [0,0,0,0]
        j = 0
        for index in range(len(label)):
            l = label[index]
            word = u[index]
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
    labels = pickle.load(open(path+"data\\questions_trn_90.pickle"))
    phrases = parse_to_phrases(questions[:90],labels)
    print ""
