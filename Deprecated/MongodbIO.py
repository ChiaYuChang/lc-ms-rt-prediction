from pymongo.errors import ConnectionFailure
from pymongo import MongoClient
from typing import Dict, Union
from ART.funcs import data_to_doc, doc_to_data

class MongodbIO():
    def __init__(
            self,
            connection_string: Union[None, str],
            database: str,
            collection: str,
            n_jobs: int = 1,
            mongodb_client_args: Dict = {}) -> None:
        self._conn_str = connection_string
        self.n_jobs = n_jobs
        self._mng_client_args = mongodb_client_args
        self._db = database
        self._cl = collection
        self._client = self.new_mongodb_client(**self._mng_client_args)

    @property
    def connection_string(self):
        return self._conn_str

    @property
    def database(self):
        self._client = self.update_connection(client=self._client)
        return self._client[self._db]

    def collection(self):
        self._client = self.update_connection(client=self._client)
        return self._client[self._db][self._cl]

    def new_mongodb_client(self, **kargs):
        if self._conn_str is None:
            return MongoClient(**kargs)
        else:
            return MongoClient(self._conn_str, **kargs)

    def test_connection(self, client: MongoClient):
        try:
            client.admin.command('ismaster')
            return True
        except ConnectionFailure:
            return False

    def update_connection(self, client: MongoClient):
        if not(self.test_connection(client=client)):
            client = self.new_mongodb_client(**self._mng_client_args)
        return client


class MongodbWriter(MongodbIO):
    def 


class MongodbReader(MongodbIO):

