import json
import urllib
from SPARQLWrapper import SPARQLWrapper, JSON

def query_freebase_entity(query, scoring, size):
    api_key = open(".api_key").read()
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'query': query,
            'key': api_key,
            'scoring': scoring,
            'lang' : 'en'
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    result = []
    for r in response['result'][:size]:
        result.append(r)
    return result

def query_freebase_property(topic_id):
    api_key = open(".api_key").read()
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

def query_fb_property_sparql(mid):
    sparql = SPARQLWrapper('http://freebase.ailao.eu:3030/freebase/query')
    sparql.setReturnFormat(JSON)
    m = mid.split('/')[-1]
    sparql.setQuery("PREFIX : <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\nSELECT ?x {\n:m." + m + " :type.object.type ?x}")
    results = sparql.query().convert()['results']['bindings']
    return [r['x']['value'][27:] for r in results]

if __name__ == "__main__":
    entity = query_freebase_entity("Beyonce",'freebase', 5)[0]
    candidates = query_fb_property_sparql(entity['mid'])
    for c in candidates:
        print c