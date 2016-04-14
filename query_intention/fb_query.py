import json
import urllib

def query_freebase(query, scoring, size, id, category=False):
    api_key = open(".api_key").read()
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'query': query,
            'key': api_key,
            'scoring': scoring,
            'lang' : 'en',
            #'type' : '/type/type',
            #'with' : 'commons'
    }
    if category:
        params['type'] = '/type/property'
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    result = None
    for r in response['result'][:size]:
        if 'id' in r.keys():
            if id in r['id']:
                result = r
                break
    return result

if __name__ == "__main__":
    print query_freebase('Marshall Hall', 'freebase', 10,'/en/marshall_hall',)