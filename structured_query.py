import pickle
from SPARQLWrapper import SPARQLWrapper, JSON

def convert_to_queries(dag, phrase):
    # queries = []
    dag[0] = '?x'
    queries = get_entity_names(dag)
    for d in range(len(dag)):
        if '?' not in dag[d]:
            dag[d] = ':' + dag[d]
    Q = ''
    for i in range(len(phrase)):
        query = ""
        index = i*(len(phrase)+1)
        if dag[index] != ':x':
            edges = dag[index+1:index+len(phrase)+1]
            for j in range(len(edges)):
                k = j*(len(phrase)+1)
                if edges[j] != ':x':
                    if edges[j] == ':SP':
                        if len(Q) == 0:
                            Q = dag[k] + " " + dag[index]
                        else:
                            Q = dag[k] + " " + Q
                            query = Q
                    elif edges[j] == ':PO':
                        if len(Q) == 0:
                            Q = dag[index] + " " + dag[k]
                        else:
                            Q = Q + " " + dag[k]
                            query = Q
                    elif edges[j] == ':SC':
                        query = dag[k] + " :type.object.type " + dag[index]
                        # query = dag[k] + " " + dag[index] + " ?cvt"
                        # dag[0] = "?cvt"
                    else:
                        query = dag[index] + " " + edges[j] + " " + dag[k]
        if len(query) > 0:
            queries.append(query)
    return queries

def create_query_file(filename, queries,phr):
    i = 0
    f = open(filename, "w")
    f.write("PREFIX : <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n")
    select = "?x"
    if ('when ',3) not in phr and ('how many ',3) not in phr:
        queries.append('?x rdfs:label ?name .\nFILTER (lang(?name) = \'en\')')
        select = "?name"
    if ('how many ',3) in phr:
        f.write("SELECT count(" + select + ") {\n")
    else:
        f.write("SELECT " + select + " {\n")
    for q in queries:
        f.write(q + ' . \n')

    f.write('}\n')
    f.close()

def get_entity_names(dag):
    alphabet = ['?a','?b','?c','?d','?e','?f','?g','?h','?i']
    index = 0
    result = []
    for i in range(len(dag)):
        d = dag[i]
        if d != 'x':
            if d[:2] == 'en':
                dag[i] = alphabet[index]
                result.append(alphabet[index]+" rdfs:label \""+d[3:].replace('_',' ').title()+"\"@en")
                index += 1
    return result

def query_fb_endpoint(query):
    sparql = SPARQLWrapper('http://freebase.ailao.eu:3030/freebase/query')
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    return sparql.query().convert()


if __name__ == "__main__":
    dags = pickle.load(open("query_gold_0_100.pickle"))
    phrases = pickle.load(open("annotate\\dags_100.pickle"))
    phr = pickle.load(open("phrases_100.pickle"))

    i = 70
    q = []
    for d, p in zip(dags[i:i+1], phrases[i:i+1]):
        q.append(convert_to_queries(d, p))
    ph = phr[i]
    i = 0
    create_query_file("test_query.txt", q[i],ph)