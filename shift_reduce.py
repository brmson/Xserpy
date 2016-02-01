from annotate.annotator import Question,object_decoder,json
import pickle

class Item(object):
    def __init__(self,stack,queue,dag,sequence):
        self.stack = stack
        self.queue = queue
        self.dag = dag
        self.sequence = sequence

def compute_score(item):
    return 1

def shift(item):
    q = item.queue[0]
    s = item.stack[:]
    s.append(q)
    return Item(s,item.queue[1:],item.dag,item.sequence+[0])

def reduce_item(item):
    s = item.stack[:]
    s.pop()
    return Item(s,item.queue,item.dag,item.sequence+[1])

def arcleft(item):
    d = [d[:] for d in item.dag]
    q = item.queue[:]
    if item.queue[0] in d[item.stack[-1]]:
        q = []
    else:
        d[item.stack[-1]] += [item.queue[0]]
    return Item(item.stack,q,d,item.sequence+[2])

def arcright(item):
    d = [dd[:] for dd in item.dag]
    q = item.queue[:]
    if item.stack[-1] in d[item.queue[0]]:
        q = []
    else:
        d[item.queue[0]] += [item.stack[-1]]
    return Item(item.stack,q,d,item.sequence+[3])

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

def derive_labels(dags,phrases):
    sequences = []
    # shift = 0
    # reduce = 1
    # arcleft = 2
    # arcright = 3
    for i in range(len(dags)):
        print i
        # sequence = []
        phrase = range(len(phrases[i]))

        ddag = dags[i]
        dag = []
        for dd in ddag:
            dag.append([int(d) for d in dd])
        new_item = Item([],phrase,[[] for p in range(len(phrase))],[])
        queue = [new_item]
        while queue:
            item = queue.pop()
            if item.dag == dag:
                print dag
                print item.sequence
                # sequences += item.sequence
                # break
            if not item.queue:
                continue
            else:
                shifted_item = [shift(item)]
                queue = shifted_item + queue
                if item.stack:
                    left_item = arcleft(item)
                    right_item = arcright(item)
                    red_item = reduce_item(item)
                    queue = [left_item,right_item,red_item] + queue

if __name__ == "__main__":

    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    questions = json.load(open(path+"data\\free917.train.examples.canonicalized.json"),object_hook=object_decoder)
    labels = pickle.load(open(path+"data\\questions_trn_90.pickle"))
    dags = pickle.load(open(path+"annotate\\dags_20.pickle"))
    phrases = parse_to_phrases(questions[:2],labels)
    derive_labels(dags[:2],phrases)
    print ""
