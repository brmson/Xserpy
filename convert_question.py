import pickle

from phrase_detection.phrase_detector import predict
from annotate.annotator import object_decoder, json
import structured_query as sq, os, argparse
from shift_reduce import shift_reduce as sr
from query_intention import query_instantiate, entity_linking as el


def convert_question(phrase, dag, qint, question, features, pos_tagged):
    phrases = []

    for U in features:
        label = predict(phrase, U, 4)
        if label == 2:
            label = 4
        phrases.append(label)
    phr, pos = sr.parse_to_phrases([question], [phrases], [pos_tagged])

    DAG = sr.shift_reduce(phr[0], pos[0], dag, 50).dag

    # examples, features = query_instantiate.create_examples([question], phr, [DAG])
    # rel_entities, simple = query_instantiate.get_db_entities([question])
    # relations, entities = query_instantiate.get_entities_relations(rel_entities)

    # intent = query_instantiate.beam_search(examples[0], 30, ['x']+relations, qint, training=False)
    intent = el.label_all(phr[0], DAG, qint, "query_intention\\ent_perceptron_trn_641.pickle", "query_intention\\edge_perceptron_trn_100.pickle")

    # structured_query.create_query_file("query.txt", structured_query.convert_to_queries(intent, phr[0]), phr)
    print sq.create_query(sq.convert_to_queries(intent, phr[0]), phr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process one question from dataset")
    parser.add_argument("fpath", help="Path to data", type=str)
    parser.add_argument("size", help="Dataset size", type=int, default=0)
    parser.add_argument("i", help="Index", type=int, default=0)
    parser.add_argument("n_iter", help="Number of iterations", type=int, default=0)
    parser.add_argument("n_iter_dag", help="Number of iterations for DAG training", type=int, default=0)
    parser.add_argument("mode", help="mode", type=str)
    args = parser.parse_args()
    size = args.size
    path = args.fpath
    i = args.i
    n_iter = args.n_iter
    n_iter_dag = args.n_iter_dag
    mode = args.mode

    sep = os.path.sep

    pos = pickle.load(open("data" + sep + "pos_tagged_" + mode + ".pickle"))
    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    features = pickle.load(open(path + "data" + sep + "phrase_detect_features_" + mode + "_" + str(size) + "_arr.pickle"))
    model_phrase = pickle.load(open(path+"models" + sep + "w_" + str(size) + "_i" + str(n_iter) + ".pickle"))
    model_dag = pickle.load(open(path+"models" + sep + "w_dag" + str(size) + "_i" + str(n_iter_dag) + ".pickle"))
    candidates = pickle.load(open(path + "data" + sep + "candidates_" + mode + "_" + str(size) + ".pickle"))
    # model_qint = pickle.load(open(path+"models" + sep + "w_qint.pickle"))

    U = sum([len(q.utterance.split()) for q in questions[:i]])
    u = len(questions[i].utterance.split())
    convert_question(model_phrase, model_dag, candidates[i], questions[i], features[U:U+u], pos[i])