import pickle
from phrase_detector import predict
from annotate.annotator import object_decoder,Question,json,parse_to_phrases
import structured_query,shift_reduce,query_instantiate

def convert_question(phrase,dag,qint,question,features,pos_tagged):
    phrases = []
    for U in features:
        phrases.append(predict(phrase,U,4))
    phr,pos = shift_reduce.parse_to_phrases([question],[phrases],[pos_tagged])
    DAG = shift_reduce.shift_reduce(phr[0],pos[0],dag,50).dag
    examples,features = query_instantiate.create_examples([question],phr,[DAG])
    rel_entities,simple = query_instantiate.get_db_entities([question])
    relations,entities = query_instantiate.get_entities_relations(rel_entities)
    intent = query_instantiate.beam_search(examples[0],30,['x']+relations,qint,training=False)
    structured_query.create_query_file("query.txt",structured_query.convert_to_queries(intent,phr[0]))

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    pos = pickle.load(open("data\\pos_tagged.pickle"))
    questions = json.load(open(path+"data\\free917.train.examples.canonicalized.json"),object_hook=object_decoder)
    features = pickle.load(open(path + "annotate\\phrase_detect_features_100_arr.pickle"))
    model_phrase = pickle.load(open(path+"models\\w_all_50.pickle"))
    model_dag = pickle.load(open(path+"models\\w_dag_all.pickle"))
    model_qint = pickle.load(open(path+"models\\w_qint.pickle"))
    i = 0
    u = len(questions[i].utterance.split())
    convert_question(model_phrase,model_dag,model_qint,questions[i],features[i:i+u],pos[i])