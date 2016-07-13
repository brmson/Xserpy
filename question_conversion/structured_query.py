"""Convert a DAG linked to a knowledge base to SPARQL query string or file"""
import pickle, sys
from SPARQLWrapper import SPARQLWrapper, JSON

def convert_to_queries(dag):
    """Convert labeled DAG to list of strings representing SPARQL WHERE conditions

    Keyword arguments:
    dag -- DAG linked to knowledge base

    """
    dct = {30: 5, 20: 4, 12: 3, 6: 2}
    queries = get_entity_names(dag)
    for d in range(len(dag)):
        if '?' not in dag[d]:
            dag[d] = ':' + dag[d]
    Q = ''
    length = dct[len(dag)]
    for i in range(length):
        query = ""
        index = i*(length+1)
        if dag[index] != ':x':
            edges = dag[index+1:index+length+1]
            for j in range(len(edges)):
                k = j*(length+1)
                if edges[j] != ':x':
                    if edges[j] == ':SP':
                        if len(Q) == 0:
                            Q = dag[k] + " " + dag[index]
                        else:
                            Q = dag[k] + " " + Q
                            query = Q
                            Q = ""
                    elif edges[j] == ':PO':
                        if len(Q) == 0:
                            Q = dag[index] + " " + dag[k]
                        else:
                            Q = Q + " " + dag[k]
                            query = Q
                            Q = ""
                    elif edges[j] == ':SC':
                        query = dag[k] + " :type.object.type " + dag[index]
                    else:
                        query = dag[index] + " " + edges[j] + " " + dag[k]
                    qs = query.split()
                    if len(query) > 0 and qs[1][0] != '?' and len(qs) == 3:
                        queries.append(query)
    return queries

def create_query_file(filename, queries, phr):
    """Write SPARQL query to a file

    Keyword arguments:
    filename -- name of output file
    queries -- list of condition strings
    phr -- words of question grouped by phrase labels

    """
    f = open(filename, "w")
    f.write("PREFIX : <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n")
    queries.append('OPTIONAL {?x rdfs:label ?name .\nFILTER (lang(?name) = \'en\')}')
    if ('how many ',3) in phr:
        f += "SELECT count(?x) {\n"
    else:
        f += "SELECT ?x ?name {\n"
    for q in queries:
        f.write(q + ' . \n')

    f.write('}\n')
    f.close()


def create_query(queries, phr):
    """Create query as a string

    Keyword arguments:
    queries -- list of condition strings
    phr -- words of question grouped by phrase labels

    """
    f = ''
    f += "PREFIX : <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    queries.append('OPTIONAL {?x rdfs:label ?name .\nFILTER (lang(?name) = \'en\')}')
    if ('how many ',3) in phr:
        f += "SELECT count(?x) {\n"
    else:
        f += "SELECT ?x ?name {\n"
    for q in queries:
        Q = q.split()
        if Q[1] != Q[2]:
            f += q + ' . \n'
        else:
            break
    f += '}\n'
    return f


def get_entity_names(dag):
    """Add conditions to determine machine ID when only human-readable ID is known

    Keyword arguments:
    dag -- DAG linked to knowledge base

    """
    alphabet = ['?a','?b','?c','?d','?e','?f','?g','?h','?i']
    index = 0
    result = []
    for i in range(len(dag)):
        d = dag[i]
        if d[:3] == 'en.':
            dag[i] = alphabet[index]
            result.append(alphabet[index]+" rdfs:label \""+d[3:].replace('_',' ').title()+"\"@en")
            index += 1
    return result

def query_fb_endpoint(query, url):
    """Send request for data to Freebase endpoint

    Keyword arguments:
    query -- query in string representation

    """
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    return sparql.query().convert()


if __name__ == "__main__":
    dags = pickle.load(open(sys.argv[2]))
    phr = pickle.load(open(sys.argv[3]))

    i = int(sys.argv[1])
    q = []
    for d in dags[i:i+1]:
        q.append(convert_to_queries(d))
    ph = phr[i]
    i = 0
    create_query_file("test_query.txt", q[i],ph)