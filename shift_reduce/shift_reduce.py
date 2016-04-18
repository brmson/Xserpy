import pickle
import os
import argparse

from annotate.annotator import object_decoder, json, examples_to_phrases, parse_dags
from phrase_detection.phrase_detector import train, init_weights, compute_score


class Item(object):
    def __init__(self, stack, queue, dag, sequence, features, data):
        self.stack = stack
        self.queue = queue
        self.dag = dag
        self.sequence = sequence
        self.data = data[:]
        self.features = features[:]
        self.features.append(self.construct_features(data[0], data[1], stack, queue, dag))

    def construct_features(self, phrase, pos, stack, queue, dag):
        features = []
        if stack:
            head = stack[-1]
            #
            features.append("ST_p."+pos[head][:-1])
            features.append("ST_w."+phrase[head][0])
            features.append("ST_p_w."+pos[head]+phrase[head][0])
        if queue:
            next = queue[0]
            features.append("N0_p."+pos[next][:-1])
            features.append("N0_w."+phrase[next][0])
            features.append("N0_p_w."+pos[next]+phrase[next][0])

            features.append("N0_t."+str(phrase[next][1]))
            features.append("N0_t_p."+str(phrase[next][1])+"_"+pos[next][:-1])
            features.append("N0_t_w."+str(phrase[next][1])+"_"+phrase[next][0])
            if stack:
                head = stack[-1]
                features.append("ST_p_w_N0_p_w."+pos[head] + phrase[head][0]+"_"+pos[next] + phrase[next][0])
                features.append("ST_p_w_N0_w_."+pos[head] + phrase[head][0]+"_"+phrase[next][0])

                features.append("ST_t_N0_p."+str(phrase[head][1])+"_"+pos[next][:-1])
                features.append("ST_w_N0_t."+phrase[head][0]+"_"+str(phrase[next][1]))

                features.append("ST_p_N0_p."+pos[head]+"_"+pos[next][:-1])
                features.append("ST_t_N0_t."+str(phrase[head][1])+"_"+str(phrase[next][1]))

                if head in dag[next]:
                    features.append("AR."+phrase[head][0]+"_"+phrase[next][0])
                if next in dag[head]:
                    features.append("AL."+phrase[next][0]+"_"+phrase[head][0])
        if len(queue) > 1:
            next = queue[1]
            features.append("N1_p."+pos[next][:-1])
            features.append("N1_w."+phrase[next][0])
            features.append("N1_p_w."+pos[next]+phrase[next][0])

            features.append("N1_t."+str(phrase[next][1]))
            features.append("N1_t_p."+str(phrase[next][1])+"_"+pos[next][:-1])
            features.append("N1_t_w."+str(phrase[next][1])+"_"+phrase[next][0])

            features.append("N0_p_N1_p."+pos[queue[0]]+"_"+pos[next][:-1])
            if len(queue) > 2:
                features.append("N0_p_N1_p_N2_p."+pos[queue[0]]+"_"+pos[next]+"_"+pos[queue[2]][:-1])
                features.append("N0_t_N1_p_N2_p."+str(phrase[queue[0]][1])+"_"+pos[next]+"_"+pos[queue[2]][:-1])
        return features

def shift(item):
    q = item.queue[0]
    s = item.stack[:]
    s.append(q)
    return Item(s, item.queue[1:], item.dag, item.sequence+[0], item.features, item.data)

def reduce_item(item):
    s = item.stack[:]
    if not s:
        return None
    s.pop()
    return Item(s, item.queue, item.dag, item.sequence+[1], item.features, item.data)

def arcleft(item):
    d = [d[:] for d in item.dag]
    q = item.queue[:]
    if not item.stack:
        return None
    elif item.queue[0] in d[item.stack[-1]]:
        return None
    else:
        d[item.stack[-1]] += [item.queue[0]]
    return Item(item.stack, q, d, item.sequence+[2], item.features, item.data)

def arcright(item):
    d = [dd[:] for dd in item.dag]
    q = item.queue[:]
    if not item.stack:
        return None
    elif item.stack[-1] in d[item.queue[0]]:
        return None
    else:
        d[item.queue[0]] += [item.stack[-1]]
    return Item(item.stack, q, d, item.sequence+[3], item.features, item.data)

def shift_reduce(sentence, pos, weights, size):
    actions = [shift, reduce_item, arcleft, arcright]
    deque = []
    start_item = type('TestItem',  (),  {})()
    start_item.queue = range(len(sentence))
    start_item.stack = []
    start_item.dag = [[] for sen in range(len(sentence))]
    start_item.features = []
    start_item.sequence = []
    start_item.data = [sentence, pos]
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
                new_score = item.score + compute_score(new_item.features[-1], weights, i)
                new_item.score = new_score
                if not new_item.queue:
                    if result is None or new_score > score:
                        result = new_item
                        score = new_score
                else:
                    lst.append(new_item)
        lst.sort(key=lambda x: x.score,  reverse=True)
        deque = lst[:size]
    return result

def check_dag(gold, dag):
    for item in zip(gold, dag):
        for i in item[1]:
            if i not in item[0]:
                return False
    return True


def parse_to_phrases(questions, labels, pos):
    pos_tag = []
    phrases = []
    for i in range(len(questions)):
        u = [q for q in questions[i].utterance.split()]
        label = [l for l in labels[i]]
        POS = [p[1] for p in pos[i]]
        j = 0
        while label[j] == 4:
            j += 1
        phrase = [u[j]+" "]
        pos_t = [POS[j]+'_']
        k = 0
        order = [label[j]]
        while j < len(label):
            if j + 1 >= len(label):
                break
            if label[j+1] == label[j]:
                phrase[k] += u[j+1] + " "
                pos_t[k] += POS[j+1] + '_'
                u.remove(u[j+1])
                POS.remove(POS[j+1])
                label.remove(label[j+1])
            else:
                j += 1
                k += 1
                while label[j] == 4:
                    j += 1
                phrase.append(u[j] + " ")
                pos_t.append(POS[j] + "_")
                order.append(label[j])
        m = 0
        while m < len(phrase):
            if order[m] == 0:
                m += 1
                continue
            n = m + 1
            while n < len(phrase):
                if order[m] == order[n]:
                    phrase[m] += phrase[n]
                    pos_t[m] += pos_t[n]
                    del phrase[n]
                    del pos_t[n]
                    del order[n]
                n += 1
            m += 1
        phrases.append(zip(phrase,order))
        pos_tag.append(pos_t)
    return phrases, pos_tag

def derive_labels(dags, phrases, pos):
    sequences = []
    seqs = []
    features = []
    # shift = 0
    # reduce = 1
    # arcleft = 2
    # arcright = 3
    for i in range(len(dags)):
        sequence = None
        if ("", 0) in phrases[i]:
            phrases[i].remove(("", 0))
        phrase = range(len(phrases[i]))
        ddag = dags[i]
        dag = []
        for dd in ddag:
            dag.append([int(d) for d in dd])
        new_item = Item([], phrase, [[] for p in range(len(phrase))], [], [], (phrases[i], pos[i]))
        queue = [new_item]

        while queue:
            item = queue.pop()
            if item.dag == dag:
                if sequence is None or len(item.sequence) < sequence:
                    sequence = item.sequence
                    feature = item.features
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

                    if left_item is not None:
                        if check_dag(dag, left_item.dag):
                            queue_item.append(left_item)
                    if right_item is not None:
                        if check_dag(dag, right_item.dag):
                            queue_item.append(right_item)

                    if red_item is not None:
                        queue_item.append(red_item)
                    queue = queue_item + queue
        if sequence is not None:
            seqs.append(sequence)
            sequences += sequence
            features += feature[:-1]
        else:
            seqs.append([])
    return zip(features, sequences),seqs

def batch_shift_reduce(sentences, pos, weights, size):
    result = []
    for s in range(len(sentences)):
        sentence = sentences[s]
        p = pos[s]
        result.append(shift_reduce(sentence, p, weights, size))
    return result

def compute_error(dags, gold):
    correct = 0
    total = 0
    noedge_err = 0
    edge_err = 0
    for i in range(len(dags)):
        dag = dags[i].dag
        g = gold[i]
        if len(dag) == len(g):
            if dag == g:
                correct += 1
                total += len(dag)**2
            else:
                for j in range(len(dag)):
                    dd = dag[j]
                    gg = g[j]
                    total += len(dag)
                    for ddd in dd:
                        if ddd not in gg:
                            noedge_err += 1
                    for ggg in gg:
                        if ggg not in dd:
                            edge_err += 1
    return noedge_err,edge_err,total,correct


if __name__ == "__main__":
    sep = os.path.sep

    parser = argparse.ArgumentParser(description="Train weights for DAG detection")
    parser.add_argument("fpath", help="Path to features and labels (array format)", type=str)
    parser.add_argument("n_iter", help="Number of iterations for training", type=int, default=0)
    parser.add_argument("--size", help="Size of dataset", type=int, default=641)
    parser.add_argument("type", help="Operating mode for script", type=str)
    parser.add_argument("mode", help="Training or testing split", type=str)
    parser.add_argument("beam", help="Beam size for beam search", type=int)
    parser.add_argument("--rate", help="Learning rate for training", type=int, default=1)
    args = parser.parse_args()
    path = args.fpath
    mode = args.mode
    size = args.size
    n_iter = args.n_iter
    beam = args.beam
    learning_rate = args.rate
    c = 4

    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    labels = pickle.load(open(path+"data" + sep + "labels_" + mode + "_" + str(size) + ".pickle"))
    labels_split = pickle.load(open(path+"data" + sep + "questions_" + mode + "_" + str(size) + ".pickle"))
    pos_tagged = pickle.load(open(path + "data" + sep + "pos_tagged_" + mode + ".pickle"))

    if 'c' in args.type:
        phr = examples_to_phrases(labels, questions)
        phrases, pos = parse_to_phrases(questions, labels_split, pos_tagged)
        DAGs = parse_dags(phrases)
        examples,seqs = derive_labels(DAGs, phrases, pos)
        empty_weights = init_weights(examples, {}, c)
        pickle.dump(examples,open(path + "data" + sep + "dag_examples_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(DAGs,open(path + "data" + sep + "gold_dags_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(seqs,open(path + "data" + sep + "gold_sequences_" + mode + "_" + str(size) + ".pickle","wb"))
        pickle.dump(empty_weights,open(path + "data" + sep + "empty_weights_dag_" + mode + "_" + str(size) + ".pickle","wb"))

    elif 'b' in args.type:
        weights = pickle.load(open(path + "models" + sep + "w_dag641_i" + str(n_iter) + ".pickle"))
        labels = pickle.load(open(path + "data" + sep + "gold_dags_" + mode + "_" + str(size) + ".pickle"))
        phrases, pos = parse_to_phrases(questions, labels_split, pos_tagged)
        d = batch_shift_reduce(phrases, pos, weights, beam)
        print compute_error(d,labels)
        
    else:
        examples = pickle.load(open(path + "data" + sep + "dag_examples_" + mode + "_" + str(size) + ".pickle"))
        empty_weights = pickle.load(open(path + "data" + sep + "empty_weights_dag_" + mode + "_" + str(size) + ".pickle"))
        weights = train(n_iter, examples, empty_weights, c, learning_rate)
        pickle.dump(weights, open(path+"models" + sep + "w_dag" + str(size) + "_i" + str(args.n_iter) + ".pickle", "wb"))
