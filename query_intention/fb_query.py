import json
import urllib

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

if __name__ == "__main__":
    entity = query_freebase_entity("Beyonce",'freebase', 5)[0]
    candidates = query_freebase_property(entity['mid'])
    for c in candidates:
        print c