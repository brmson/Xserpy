import pickle

def convert_to_query():
    return

def create_query_file(filename,entities):
    # alphabet = ["a","b","c","d","e","f","g"]
    # dic = {}
    i = 0
    f = open(filename,"w")
    f.write("prefix : <http://rdf.basekb.com/ns/>\nselect ?a {\n "+ entities[1] + " " + entities[0] + " ?a .\n}")
    f.close()

if __name__ == "__main__":
    create_query_file("test_query.txt",["fb:comic_books.comic_book_issue.issue_number","fb:en.a_dream_of_a_thousand_cats"])