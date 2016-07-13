"""Obtain information from Freebase"""
import json, urllib, sys, os
from SPARQLWrapper import SPARQLWrapper, JSON

PATH_TO_KEY = os.getcwd() + os.path.sep + ".api_key"

def query_kg_entity(query, size):
    """Query Knowledge Graph for entity candidates

    Keyword arguments:
    query -- natural language text query
    size -- number of candidates returned

    """
    api_key = open(PATH_TO_KEY).read()
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
            'query': query,
            'key': api_key,
            'languages' : 'en,cs',
            'limit': size
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    result = []
    for r in response['itemListElement']:
        result.append(r['result'])
    return result

def query_freebase_entity(query, scoring, size):
    """Query Freebase for entity candidates

    Keyword arguments:
    query -- natural language text query
    scoring -- type of scoring
    size -- number of candidates returned

    """
    api_key = open(PATH_TO_KEY).read()
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'query': query,
            'key': api_key,
            'scoring': scoring,
            'lang': 'en'
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    result = []
    for r in response['result'][:size]:
        result.append(r)
    return result

def query_freebase_property(topic_id):
    """Obtain properties of one entity through Google API

    Keyword arguments:
    topic_id -- machine ID of desired entity

    """
    api_key = open(PATH_TO_KEY).read()
    service_url = 'https://www.googleapis.com/freebase/v1/topic'
    params = {
      'key': api_key,
      'filter': 'allproperties'
    }
    url = service_url + topic_id + '?' + urllib.urlencode(params)
    topic = json.loads(urllib.urlopen(url).read())
    result = []
    for property in topic['property']:
        if property[:5] != '/type':
            result.append(property)
    return result

def query_fb_property_sparql(mid, url):
    """Obtain properties of one entity through SPARQL

    Keyword arguments:
    mid -- machine ID of desired entity
    url -- URL of Freebase endpoint

    """
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    m = mid.split('/')[-1]
    sparql.setQuery("PREFIX : <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nSELECT ?x {\n:m." + m + " :type.object.type ?x}")
    results = sparql.query().convert()['results']['bindings']
    return [r['x']['value'][27:] for r in results]

def get_qid_sparql(mid, url):
    """Convert Freebase MID to Wikidata QID

    Keyword arguments:
    mid -- Freebase machine ID
    url -- URL of Wikidata SPARQL endpoint

    """
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    query = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT * WHERE {   	?a wdt:P646 \"" + mid + "\" 	}"
    sparql.setQuery(query)
    results = sparql.query().convert()['results']['bindings']
    return results[0]['a']['value'][31:]

if __name__ == "__main__":
    url = sys.argv[2]
    # entity = query_freebase_entity(sys.argv[1],'freebase', 5)
    # for c in entity:
    #     print c
    entity = query_kg_entity(sys.argv[1], 5)
    # candidates = query_fb_property_sparql(entity['mid'])
    url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
    for c in entity:
        print get_qid_sparql(c['@id'][3:], url)