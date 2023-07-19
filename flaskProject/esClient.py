from elasticsearch import Elasticsearch


class ESClass:
    def __init__(self):
        self.client = Elasticsearch(hosts='http://localhost:9200')
        # self.client = Elasticsearch(hosts='http://121.43.150.228:9200')
        print("Elasticsearch client is set")


es = ESClass()
