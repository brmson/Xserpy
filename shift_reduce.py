from annotate.annotator import Question,object_decoder,json
from phrase_detector import train,predict,init_weights,compute_score
import pickle,time

class Item(object):
    def __init__(self,stack,queue,dag,sequence,features,data):
        self.stack = stack
        self.queue = queue
        self.dag = dag
        self.sequence = sequence
        self.data = data[:]
        self.features = features[:]
        self.features.append(self.construct_features(data[0],data[1],stack,queue))

    def construct_features(self,phrase,pos,stack,queue):
        features = []
        if stack:
            head = stack[-1]
            features.append(pos[head][:-1])
            features.append("ST_w_"+phrase[head][0])
            features.append("ST_p_w_"+pos[head]+phrase[head][0])
        if queue:
            next = queue[0]
            features.append(pos[next][:-1])
            features.append("N0_w_"+phrase[next][0])
            features.append("N0_p_w_"+pos[next]+phrase[next][0])
            features.append("N0_t_"+str(phrase[next][1]))
            features.append("N0_t_p_"+str(phrase[next][1])+"_"+pos[next])
            features.append("N0_t_w_"+str(phrase[next][1])+"_"+phrase[next][0])
            if stack:
                head = stack[-1]
                features.append("ST_p_w_"+pos[head]+phrase[head][0]+"_"+"N0_p_w_"+pos[next]+phrase[next][0])
                features.append("ST_p_w_"+pos[head]+phrase[head][0]+"_"+"N0_w_"+phrase[next][0])

        return features

def shift(item):
    q = item.queue[0]
    s = item.stack[:]
    s.append(q)
    return Item(s,item.queue[1:],item.dag,item.sequence+[0],item.features,item.data)

def reduce_item(item):
    s = item.stack[:]
    if not s:
        return None
    s.pop()
    return Item(s,item.queue,item.dag,item.sequence+[1],item.features,item.data)

def arcleft(item):
    d = [d[:] for d in item.dag]
    q = item.queue[:]
    if not item.stack:
        return None
    elif item.queue[0] in d[item.stack[-1]]:
        return None
    else:
        d[item.stack[-1]] += [item.queue[0]]
    return Item(item.stack,q,d,item.sequence+[2],item.features,item.data)

def arcright(item):
    d = [dd[:] for dd in item.dag]
    q = item.queue[:]
    if not item.stack:
        return None
    elif item.stack[-1] in d[item.queue[0]]:
        return None
    else:
        d[item.queue[0]] += [item.stack[-1]]
    return Item(item.stack,q,d,item.sequence+[3],item.features,item.data)

def shift_reduce(sentence,pos,weights,size):
    actions = [shift,reduce_item,arcleft,arcright]
    deque = []
    start_item = type('TestItem', (), {})() # Item([],sentence,False,[],[],[])
    start_item.queue = range(len(sentence))
    start_item.stack = []
    start_item.dag = [[] for sen in range(len(sentence))]
    start_item.features = []
    start_item.sequence = []
    start_item.data = [sentence,pos]
    start_item = shift(start_item)
    start_item.score = 0
    deque.append(start_item)
    result = None
    score = 0
    while deque:
        lst = []
        for item in deque:
            for i in range(len(actions)):
                new_item = actions[i](item)
                if new_item is None:
                    continue
                new_score = item.score + compute_score(new_item.features[-1],weights,i)
                new_item.score = new_score
                if not new_item.queue:
                    if result is None or new_score > score:
                        result = new_item
                        score = new_score
                else:
                    lst.append(new_item)
        lst.sort(key=lambda x: x.score, reverse=True)
        deque = lst[:size]
    return result

def check_dag(gold,dag):
    for item in zip(gold,dag):
        for i in item[1]:
            if i not in item[0]:
                return False
    return True

def parse_to_phrases(questions,labels,pos_tagged):
    phrases = []
    pos_phrases = []
    for i in range(len(questions)):
        u = questions[i].utterance.split()
        label = labels[i]
        pos = pos_tagged[i]
        dic = {}
        phrase = ["","","",""]
        pos_phrase = ["","","",""]
        order = [0,0,0,0]
        j = 0
        for index in range(len(label)):
            l = label[index]
            word = u[index]
            p = pos[index]
            index += 1
            if l == 4:
                continue
            if l not in dic.keys():
                dic[l] = j
                order[j] = l
                j += 1
            phrase[dic[l]] += word + " "
            pos_phrase[dic[l]] += p[1]+"_"
        phrases.append(zip(phrase,order))
        pos_phrases.append(pos_phrase)
    return phrases,pos_phrases

def derive_labels(dags,phrases,pos):
    sequences = []
    features = []
    # shift = 0
    # reduce = 1
    # arcleft = 2
    # arcright = 3
    for i in range(len(dags)):
        # print i
        sequence = None
        if ("",0) in phrases[i]:
            phrases[i].remove(("",0))
        phrase = range(len(phrases[i]))
        ddag = dags[i]
        dag = []
        for dd in ddag:
            dag.append([int(d) for d in dd])
        # print dag
        new_item = Item([],phrase,[[] for p in range(len(phrase))],[],[],(phrases[i],pos[i]))
        queue = [new_item]

        while queue:
            item = queue.pop()
            if item.dag == dag:
                if sequence is None or len(item.sequence) < sequence:
                # print dag
                # print item.sequence
                    sequence = item.sequence
                    feature = item.features
                # break
            if not item.queue:
                continue
            else:
                shifted_item = [shift(item)]
                queue = shifted_item + queue
                if item.stack:
                    queue_item = []
                    left_item = arcleft(item)
                    right_item = arcright(item)
                    red_item = reduce_item(item)

                    if check_dag(dag,left_item.dag):
                        queue_item.append(left_item)

                    if check_dag(dag,right_item.dag):
                        queue_item.append(right_item)

                    # if check_dag(dag,red_item.dag):
                    queue_item.append(red_item)
                    # queue_item = [left_item,right_item,red_item]
                    queue = queue_item + queue
        if sequence is not None:
            sequences += sequence
            features += feature[:-1]
    return zip(features,sequences)

def batch_shift_reduce(sentences,pos,weights,size):
    result = []
    for s in range(len(sentences)):
        sentence = sentences[s]
        p = pos[s]
        result.append(shift_reduce(sentence,p,weights,size))
    return result

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    questions = json.load(open(path+"data\\free917.train.examples.canonicalized.json"),object_hook=object_decoder)
    labels = pickle.load(open(path+"data\\questions_trn_100.pickle"))
    dags = pickle.load(open(path+"annotate\\dags_100.pickle"))
    pos_tagged = pickle.load(open(path + "data\\pos_tagged.pickle"))
    i = 10
    phrases,pos = parse_to_phrases(questions[:i],labels[:i],pos_tagged[:i])
    start = time.time()
    examples = derive_labels(dags[:i],phrases,pos)
    c = 4
    size = 20
    weights = train(50,examples,init_weights(examples,{},c),c)
    dags = batch_shift_reduce(phrases,pos,weights,size)
    print time.time()-start
