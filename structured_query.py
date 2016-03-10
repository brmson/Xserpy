import pickle

def convert_to_queries(dag,phrase):
    queries = []
    dag[0] = '?x'
    for d in range(1,len(dag)):
        dag[d] = ':' + dag[d]
    for i in range(len(phrase)):
        query = ""
        index = i*len(phrase)
        edges = dag[index+1:index+len(phrase)+1]
        for j in range(len(edges)):
            k = j*len(phrase)
            if edges[j] != ':x':
                if edges[j] == ':SP':
                    pass
                elif edges[j] == ':PO':
                    pass
                elif edges[j] == ':SC':
                    query = dag[index]+" object.type "+dag[k]
                else:
                    query = dag[index] + " " + edges[j] + " " + dag[k]
        if len(query) > 0:
            queries.append(query)
    return queries

def create_query_file(filename,queries):
    i = 0
    f = open(filename,"w")
    f.write("PREFIX : <http://rdf.basekb.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nSELECT ?name {\n")
    for q in queries:
        f.write(q + ' . \n')
    f.write('?year rdfs:label ?name .\nFILTER (lang(?name)= \'en\')\n}\n')
    f.close()

if __name__ == "__main__":
    dags = pickle.load(open("query_int_20.pickle"))
    phrases = pickle.load(open("annotate\\dags_100.pickle"))
    q = []
    for d,p in zip(dags,phrases):
        q.append(convert_to_queries(d,p))
    create_query_file("test_query.txt",q[15])