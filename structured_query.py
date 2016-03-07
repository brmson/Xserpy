import pickle

def convert_to_queries(dag,phrase):
    queries = []
    dag[0] = '?x'
    for i in range(len(phrase)):
        query = ""
        index = i*len(phrase)
        edges = dag[index+1:index+len(phrase)+1]
        for j in range(len(edges)):
            k = j*len(phrase)
            if edges[j] != 'x':
                if edges[j] == 'SP':
                    pass
                elif edges[j] == 'PO':
                    pass
                elif edges[j] == 'SC':
                    query = dag[index]+" object.type "+dag[k]
                else:
                    query = dag[index] + " " + edges[j] + " " + dag[k]
        queries.append(query)
    return queries

def create_query_file(filename,entities):
    # alphabet = ["a","b","c","d","e","f","g"]
    # dic = {}
    i = 0
    f = open(filename,"w")
    f.write("prefix : <http://rdf.basekb.com/ns/>\nselect ?a {\n "+ entities[1][2:] + " " + entities[0][2:] + " ?a .\n}")
    f.close()

if __name__ == "__main__":
    dags = pickle.load(open("query_gold_10.pickle"))
    phrases = pickle.load(open("annotate\\dags_100.pickle"))
    for d,p in zip(dags,phrases):
        print convert_to_queries(d,p)
    create_query_file("test_query.txt",["fb:comic_books.comic_book_issue.issue_number","fb:en.a_dream_of_a_thousand_cats"])