import pickle

from phrase_detection.phrase_detector import predict
from phrase_detection.feature_constructor import pos_tag, ner_tag, Question, construct_feature
from annotate.annotator import object_decoder, json
import structured_query as sq, os, argparse
from shift_reduce import shift_reduce as sr
from query_intention import query_instantiate, entity_linking as el


def get_phrases_free(question):
    pd_model = pickle.load(open("C:\\Users\\Martin\\PycharmProjects\\xserpy\\models\\w_641_i20.pickle"))

    q = Question(question, "")
    utterance = question.split()

    labels = []
    pos = pos_tag([q])
    ner = ner_tag([q],"C:\\Users\\Martin\\PycharmProjects\\xserpy\\stanford-nlp\\", "C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe")
    l = 4

    u = ["", ""] + utterance + ["", ""]
    p = ['', ''] + [pp[1] for pp in pos[0]] + ['', '']
    n = ['', ''] + [nn[1] for nn in ner[0]] + ['', '']

    for j in range(2, len(u)-2):
        feature = construct_feature(p, u, n, j, l)
        label = predict(pd_model, feature, 4)
        labels.append(label)
        l = label
    return labels, pos, q

def get_phrases(phrase, features):
    phrases = []

    for U in features:
        label = predict(phrase, U, 4)
        if label == 2:
            label = 4
        phrases.append(label)
    return phrases

def convert_question(dag, candidates, question, labels, pos_tagged, filename):

    phr, pos = sr.parse_to_phrases([question], [labels], [pos_tagged])
    DAG = sr.shift_reduce(phr[0], pos[0], dag, 50).dag

    ent_path = "query_intention\\ent_lr_trn_641.pickle"
    edge_path = "query_intention\\edge_lr_trn.pickle"
    rel_path = "query_intention\\relation_lr_trn_641.pickle"
    bow_path = "query_intention\\bow_dict_all.pickle"
    g_path = "query_intention\\rel_dict.pickle"

    intent = el.label_all(phr[0], DAG, candidates, ent_path, edge_path, rel_path, bow_path, g_path)
    queries = sq.convert_to_queries(intent)
    # sq.create_query_file(filename, queries, phr[0])
    if len(queries) > 0:
        query = "" # sq.create_query(queries, phr[0])
        sq.create_query_file(filename, queries, phr[0])
    else:
        query = "SELECT ?a WHERE {}"

    return query # sq.query_fb_endpoint(query)

def process_answer(answer, gold_answer):
    partial = False
    correct = False

    vrs = answer['head']['vars']
    bindings = answer['results']['bindings']
    # datatypes = [a[vrs[0]]['datatype'] for a in bindings]
    if len(bindings) == 0:
        if gold_answer[0] == '\n ':
            return True
        else:
            return False
    if len(bindings[0].keys()) == 0:
        if gold_answer[0] == '\n ':
            return True
        else:
            return False
    types = [a[vrs[0]]['type'] for a in bindings]
    values = [a[vrs[0]]['value'] for a in bindings]

    gold = [g[1:-3].split() for g in gold_answer]
    for gg in gold:
        for G in gg:
            if G[0] == "\"":
                g_index = gg.index(G)
                for GG in gg[g_index+1:]:
                    gg[g_index] += " " + GG
                    gg.remove(GG)
    c = 0
    for value in values:
        if ' ' in value:
            value = "\"" + value + "\""
        for g in gold:
            if g[0] == 'date':
                v = value.split('-')
                if g[2][0] == '-':
                    partial = v[0] == g[1]
                else:
                    partial = v == g[1:]
                c += int(partial)
            elif value in g:
                partial = True
                c += 1
        if c == len(values):
            correct = True
    return correct or partial

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process one question from dataset")
    parser.add_argument("fpath", help="Path to data", type=str)
    parser.add_argument("size", help="Dataset size", type=int, default=0)
    parser.add_argument("i", help="Index", type=int, default=0)
    parser.add_argument("n_iter", help="Number of iterations", type=int, default=0)
    parser.add_argument("n_iter_dag", help="Number of iterations for DAG training", type=int, default=0)
    parser.add_argument("mode", help="mode", type=str)
    parser.add_argument("type", help="type", type=str)
    args = parser.parse_args()
    size = args.size
    path = args.fpath
    i = args.i
    n_iter = args.n_iter
    n_iter_dag = args.n_iter_dag
    mode = args.mode
    type = args.type

    sep = os.path.sep

    pos = pickle.load(open("data" + sep + "pos_tagged_" + mode + ".pickle"))
    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    features = pickle.load(open(path + "data" + sep + "phrase_detect_features_" + mode + "_" + str(size) + "_arr.pickle"))
    model_phrase = pickle.load(open(path+"models" + sep + "w_641_i" + str(n_iter) + ".pickle"))
    model_dag = pickle.load(open(path+"models" + sep + "w_dag641_i" + str(n_iter_dag) + ".pickle"))
    candidates = pickle.load(open(path + "data" + sep + "candidates_" + mode + "_" + str(size) + ".pickle"))
    gold_answers = [(line + " ").split(') ') for line in open('data' + sep + 'free917_' + mode +'_answers.txt')]
    correct = 0

    if 'i' in type:
        question = raw_input("Enter question: ")
        phrases, pos, q = get_phrases_free(question)
        answer = convert_question(model_dag, candidates[i], q, phrases, pos[0], 'queries' + sep + mode + "_" + str(i+1)+".sparql")

    elif 'f' in type:
        questions = [line.strip() for line in ""]
        for q in questions:
            phrases, pos, q = get_phrases_free(q)
            answer = convert_question(model_dag, candidates[i], q, phrases, pos[0], 'queries' + sep + mode + "_" + str(i+1)+".sparql")

    else:
        I = []# [2, 26, 126, 191, 244]
        for i in range(len(questions)):
            print str(i+1)
            if i in I:
                continue
            U = sum([len(q.utterance.split()) for q in questions[:i]])
            u = len(questions[i].utterance.split())
            try:
                phrases = get_phrases(model_phrase, features[U:U+u])
                answer = convert_question(model_dag, candidates[i], questions[i], phrases, pos[i], 'queries' + sep + mode + "_" + str(i+1)+".sparql")
                # if process_answer(answer, gold_answers[i]):
                #     # print gold_answers[i]
                #     correct += 1
                #     print str(correct) + "/" + str(i+1)
            except Exception, e:
                print repr(e)
    print correct # 66 tst