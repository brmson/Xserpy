from flask import Flask, abort, request, Response
from convert_question import *
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/ask', methods=['GET'])
def get_task():
    question = request.args.get('q')
    print question
    if len(question) == 0:
        abort(404)
    phrases, pos, q, candidates = get_phrases_free(question, model_path, nlp_path, java_path)
    answer = convert_question(model_dag, candidates[0], q, phrases, pos[0], 'queries' + sep + mode + "_0.sparql", path + "query_intention\\")
    answers = convert_answer(answer)
    if len(answers.keys()) == 0:
        return "Answer not found"
    return Response(create_table(answers), mimetype="text/plain")

def create_table(answers):
    table = ""
    v = '.1' if '.1' in answers.keys() else 'x' if len(answers['name']) == 0 else 'name'
    length = max(11, max([len(A) for A in answers[v]]) + 5)
    table += '-'*length + '\n'
    table += '|' + 'Answer'.rjust(length-3, ' ') + ' |' + '\n'
    table += '-'*length + '\n'
    for a in answers[v]:
        table += '|' + a.rjust(length-3, ' ') + ' |' + '\n'
    table += '-'*length
    return table

if __name__ == '__main__':
    size = 276
    global path, java_path, nlp_path
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    java_path = "C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe"
    nlp_path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\stanford-nlp\\"
    n_iter = 50
    n_iter_dag = 20
    mode = "tst"
    type = "i"

    sep = os.path.sep

    global model_path, pos, questions, features, model_phrase, model_dag, candidates, gold_answers
    model_path = path + "models" + sep + "w_641_i" + str(n_iter) + "_0.1.pickle"
    pos = pickle.load(open(path + "data" + sep + "pos_tagged_" + mode + ".pickle"))
    questions = json.load(open(path+"data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    features = pickle.load(open(path + "data" + sep + "phrase_detect_features_" + mode + "_" + str(size) + "_arr.pickle"))
    model_phrase = pickle.load(open(model_path))
    model_dag = pickle.load(open(path+"models" + sep + "w_dag641_i" + str(n_iter_dag) + ".pickle"))
    gold_answers = [(line + " ").split(') ') for line in open(path + 'data' + sep + 'free917_' + mode +'_answers.txt')]
    app.run(debug=True)