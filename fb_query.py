import json
import urllib

def query_freebase(query,scoring,size,category=False):
    api_key = open(".api_key").read()
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'query': query,
            'key': api_key,
            'scoring': scoring,
            'lang' : 'en'
    }
    if category:
        params['type'] = '/type/type'
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    return response['result'][:size]

if __name__ == "__main__":
    print query_freebase('cause of death','freebase',5)