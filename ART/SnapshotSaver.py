import json

from numpy import isin

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.results import UpdateResult, InsertOneResult, InsertManyResult
from typing import Callable, Dict, NamedTuple, Optional
from ART.funcs import json_snapshot_to_doc, doc_to_json_snapshot
from time import sleep

class MongoDB():

    def __init__(self, path_to_auth_json: str, snapshot_id: Optional[ObjectId] = None):
        with open(path_to_auth_json) as f:
            db_auth = json.load(f)
        self._username = db_auth["username"]
        self._password = db_auth["password"]
        self._hosts = db_auth["hosts"]
        self._replica_name = db_auth["replicaName"]
        self._auth_db = db_auth["authenticationDatabase"]

        if "db" in db_auth.keys():
            self._db = db_auth["db"]
        else:
            self._db = None
        
        self._snapshot_id = snapshot_id

    @property
    def username(self) -> str:
        return self._username
    
    @property
    def password(self) -> str:
        return self._password

    @property
    def auth_db(self) -> str:
        return self._auth_db

    @property
    def replica_name(self) -> str:
        return self._replica_name

    @property
    def hosts(self) -> str:
        return ",".join(
            [host["host"] + ":" + str(host["port"])
                for host in self._hosts]
        )
    @property
    def snapshot_id(self) -> ObjectId:
        return self._snapshot_id

    @snapshot_id.setter
    def snapshot_id(self, obj_id: ObjectId):
        self._snapshot_id = obj_id

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, dbs: Dict):
        self._db = dbs
    
    @property
    def uri(self) -> str:
        uri = f"mongodb://{self.username}:{self.password}@{self.hosts}/" +\
            f"?authSource={self.auth_db}" +\
            f"&replicaSet={self.replica_name}&readPreference=primary"
        return uri

    def _rubust_db_IO(
            self, func: Callable,
            max_retry: int = 10,
            sleep_time: float = 3.0,
            **kargs):
        mongodb_opr_rslt = func(**kargs)

        # for query
        if isinstance(mongodb_opr_rslt, dict) or mongodb_opr_rslt is None:
            return mongodb_opr_rslt

        # for insertion, replacement
        retry = 0
        while not(mongodb_opr_rslt.acknowledged):
            mongodb_opr_rslt = func(**kargs)
            retry += 1
            sleep(sleep_time)
            if retry >= max_retry:
                raise IOError("Could not complete MongoDB operation.")

        if isinstance(mongodb_opr_rslt, UpdateResult):
            return mongodb_opr_rslt.raw_result
        elif isinstance(mongodb_opr_rslt, InsertOneResult):
            return mongodb_opr_rslt.inserted_id
        elif isinstance(mongodb_opr_rslt, InsertManyResult):
            return mongodb_opr_rslt.inserted_ids
        else:
            return None

    def save_snapshot(self, snapshot):
        doc_snapshot = json_snapshot_to_doc(snapshot)
        with MongoClient(self.uri) as client:
            mongo_col = client[self.db["db"]][self.db["col"]]
            if self._snapshot_id is None:
                return self.insert_snapshot(
                    mongo_col=mongo_col,
                    doc=doc_snapshot)
            else:                
                return self.update_snapshot(
                    mongo_col=mongo_col,
                    query=({"_id": self._snapshot_id}),
                    doc=doc_snapshot)

    def update_snapshot(self, mongo_col: Collection, query: Dict, doc: Dict): 
        raw_result = self._rubust_db_IO(
            func=mongo_col.replace_one,
            filter=query,
            replacement=doc
        )
        return raw_result

    def insert_snapshot(self, mongo_col: Collection, doc: Dict):
        self._snapshot_id = self._rubust_db_IO(
            func=mongo_col.insert_one,
            document=doc
        )
        return self._snapshot_id

    def read_snapshot(self, query: Optional[Dict] = None):
        with MongoClient(self.uri) as client:
            mongo_col = client[self.db["db"]][self.db["col"]]
            
            if query is None:
                if self._snapshot_id is None:
                    raise ValueError("query and snapshot_id are both None.")
                else:
                    query = {"_id": self._snapshot_id}
            doc = self._rubust_db_IO(
                func=mongo_col.find_one,
                filter=query
            )
        if doc is None:
            return (None, None)
        else:
            return (doc["_id"], doc_to_json_snapshot(doc))
