import pickle

from phrase_detection.phrase_detector import predict
from phrase_detection.feature_constructor import pos_tag, ner_tag, Question, construct_feature
from annotate.annotator import object_decoder, json
import structured_query as sq, os, argparse
from shift_reduce import shift_reduce as sr
from query_intention import query_instantiate, entity_linking as el


def get_phrases_free(question, model, nlp_path, java_path):
    """Convert a string question to list of phrases

    Keyword arguments:
    question -- question in string form
    model -- model for phrase detection
    nlp_path -- path to Stanford NLP tagger
    java_path -- path to Java instalation

    """
    pd_model = pickle.load(open(model))

    q = Question(question, "")
    utterance = question.split()

    labels = []
    pos = pos_tag([q])
    ner = ner_tag([q], nlp_path, java_path)
    l = 4

    u = ["", ""] + utterance + ["", ""]
    p = ['', ''] + [pp[1] for pp in pos[0]] + ['', '']
    n = ['', ''] + [nn[1] for nn in ner[0]] + ['', '']

    for j in range(2, len(u)-2):
        feature = construct_feature(p, u, n, j, l)
        label = predict(pd_model, feature, 4)
        labels.append(label)
        l = label
    phr, pos_t = sr.parse_to_phrases([q], [labels], pos)
    candidates = el.obtain_entity_candidates(phr, 5)
    return labels, pos, q, candidates

def get_phrases(phrase, features):
    """Convert a question from the dataset to list of phrases

    Keyword arguments:
    question -- question in object form
    features -- features for phrase detection

    """
    phrases = []

    for U in features:
        label = predict(phrase, U, 4)
        if label == 2:
            label = 4
        phrases.append(label)
    return phrases

def convert_question(dag, candidates, question, labels, pos_tagged, filename, path):
    """Convert a question from dataset to list of phrases

    Keyword arguments:
    dag -- model for dag parsing
    candidates -- list of lists of entity candidates
    question -- question in object form
    labels -- list of lists phrase tags
    pos_tagged -- list of lists of POS tags
    filename -- output file name
    path -- path to files

    """
    phr, pos = sr.parse_to_phrases([question], [labels], [pos_tagged])
    DAG = sr.shift_reduce(phr[0], pos[0], dag, 50).dag

    ent_path = path + "ent_lr_trn_641.pickle"
    edge_path = path + "edge_lr_trn.pickle"
    rel_path = path + "relation_lr_trn_641.pickle"
    bow_path = path + "bow_dict_all.pickle"
    g_path = path + "rel_dict.pickle"
    dct_path = path + "edge_dict.pickle"
    a_path = path + "bow_all_words_dict.pickle"

    intent = el.label_all(phr[0], DAG, candidates, ent_path, edge_path, rel_path, bow_path, g_path, dct_path, a_path)
    queries = sq.convert_to_queries(intent)
    if len(queries) > 0:
        # sq.create_query_file(filename, queries, phr[0])
        query = sq.create_query(queries, phr[0])
    else:
        query = "SELECT ?a WHERE {}"

    return sq.query_fb_endpoint(query)

def process_answer(answer, gold_answer):
    """Check correctness of response

    Keyword arguments:
    answer -- response returned from KB
    gold_answer -- correct answer

    """
    partial = False
    correct = False

    values = convert_answer(answer)
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
                v = []
                for V in value.split('-'):
                    if V[0] == '0':
                        v.append(V[1:])
                    else:
                        v.append(V)
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

def convert_answer(answer):
    vrs = answer['head']['vars']
    bindings = answer['results']['bindings']
    if len(bindings) == 0:
        return []
    if len(bindings[0].keys()) == 0:
        return []
    values = []
    for a in bindings:
        for v in vrs:
            if v in a.keys():
                values += [a[v]['value'] for a in bindings]
            else:
                values += []
    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one question from dataset")
    parser.add_argument("fpath", help="Path to data", type=str)
    parser.add_argument("size", help="Dataset size", type=int, default=0)
    parser.add_argument("i", help="Index", type=int, default=0)
    parser.add_argument("n_iter", help="Number of iterations", type=int, default=0)
    parser.add_argument("n_iter_dag", help="Number of iterations for DAG training", type=int, default=0)
    parser.add_argument("mode", help="mode", type=str)
    parser.add_argument("type", help="type", type=str)
    parser.add_argument("java_path", help="Path to Java", type=str)
    parser.add_argument("nlp_path", help="Path to Stanford NLP", type=str)
    args = parser.parse_args()
    size = args.size
    path = args.fpath
    java_path = args.java_path
    nlp_path = args.nlp_path
    i = args.i
    n_iter = args.n_iter
    n_iter_dag = args.n_iter_dag
    mode = args.mode
    type = args.type

    sep = os.path.sep
    model_path = path + "models" + sep + "w_641_i50_0.1.pickle"

    pos = pickle.load(open("data" + sep + "pos_tagged_" + mode + ".pickle"))
    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    features = pickle.load(open(path + "data" + sep + "phrase_detect_features_" + mode + "_" + str(size) + "_arr.pickle"))
    model_phrase = pickle.load(open(model_path))
    model_dag = pickle.load(open(path+"models" + sep + "w_dag641_i20.pickle"))
    candidates = pickle.load(open(path + "data" + sep + "candidates_" + mode + "_" + str(size) + ".pickle"))
    gold_answers = [(line + " ").split(') ') for line in open('data' + sep + 'free917_' + mode +'_answers.txt')]
    correct = 0

    if 'i' in type:
        question = raw_input("Enter question: ")
        if question[-1] == '?':
            question = question[:-1]
        phrases, pos, q, candidates = get_phrases_free(question, model_path, nlp_path, java_path)
        answer = convert_question(model_dag, candidates[0], q, phrases, pos[0], 'queries' + sep + mode + "_" + str(i+1)+".sparql", "query_intention\\")
        print convert_answer(answer)

    elif 'f' in type:
        questions = [line.strip() for line in ""]
        for q in questions:
            phrases, pos, q = get_phrases_free(q)
            answer = convert_question(model_dag, candidates[i], q, phrases, pos[0], 'queries' + sep + mode + "_" + str(i+1)+".sparql")
            print convert_answer(answer)

    else:
        for i in range(len(questions)):
            print str(i+1)
            U = sum([len(q.utterance.split()) for q in questions[:i]])
            u = len(questions[i].utterance.split())
            try:
                phrases = get_phrases(model_phrase, features[U:U+u])
                answer = convert_question(model_dag, candidates[i], questions[i], phrases, pos[i], 'queries' + sep + mode + "_" + str(i+1)+".sparql", path + "query_intention" + sep)
                if process_answer(answer, gold_answers[i]):
                    print gold_answers[i]
                    correct += 1
            except Exception, e:
                print repr(e)
    print correct