import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm

def set_data_directory(data_dir: str = "data"):
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    os.environ['IR_DATASETS_HOME'] = str(data_path)
    return data_path

set_data_directory("data")

import ir_datasets


class DatasetLoader:
    def __init__(self, dataset_name: str = "msmarco-passage/trec-dl-2019/judged", data_dir: str = "data"):
        set_data_directory(data_dir)
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset = ir_datasets.load(dataset_name)
        self._documents_cache = None
        self._queries_cache = None
        self._qrels_cache = None
    
    def load_documents(self, limit: Optional[int] = None, include_relevant_doc_ids: Optional[Set[str]] = None) -> Tuple[List[str], Dict[str, int]]:
        if self._documents_cache is not None and include_relevant_doc_ids is None:
            docs, doc_index = self._documents_cache
            if limit:
                limited_docs = docs[:limit]
                limited_index = {doc_id: idx for doc_id, idx in doc_index.items() if idx < limit}
                return limited_docs, limited_index
            return docs, doc_index
        
        documents = []
        doc_index = {}
        doc_iter = self.dataset.docs_iter()
        
        if limit:
            doc_iter = tqdm(doc_iter, desc="Loading documents", total=limit)
            for i, doc in enumerate(doc_iter):
                if i >= limit:
                    break
                documents.append(doc.text)
                doc_index[doc.doc_id] = i
        else:
            doc_iter = tqdm(doc_iter, desc="Loading documents")
            for i, doc in enumerate(doc_iter):
                documents.append(doc.text)
                doc_index[doc.doc_id] = i
        
        if include_relevant_doc_ids:
            docs_store = self.dataset.docs_store()
            for doc_id in tqdm(include_relevant_doc_ids, desc="Loading relevant documents"):
                if doc_id not in doc_index:
                    try:
                        doc = docs_store.get(doc_id)
                        if doc:
                            idx = len(documents)
                            documents.append(doc.text)
                            doc_index[doc_id] = idx
                    except:
                        pass
        
        if include_relevant_doc_ids is None:
            self._documents_cache = (documents, doc_index)
        return documents, doc_index
    
    def load_queries(self, limit: Optional[int] = None) -> Dict[str, str]:
        if self._queries_cache is not None:
            items = list(self._queries_cache.items())
            return dict(items[:limit]) if limit else self._queries_cache
        
        queries = {}
        query_iter = self.dataset.queries_iter()
        
        if limit:
            query_iter = tqdm(query_iter, desc="Loading queries", total=limit)
            for i, query in enumerate(query_iter):
                if i >= limit:
                    break
                queries[query.query_id] = query.text
        else:
            query_iter = tqdm(query_iter, desc="Loading queries")
            for query in query_iter:
                queries[query.query_id] = query.text
        
        self._queries_cache = queries
        return queries
    
    def load_qrels(self) -> Dict[str, Set[str]]:
        if self._qrels_cache is not None:
            return self._qrels_cache
        
        qrels = {}
        qrel_iter = self.dataset.qrels_iter()
        
        for qrel in tqdm(qrel_iter, desc="Loading qrels"):
            query_id = qrel.query_id
            doc_id = qrel.doc_id
            
            if query_id not in qrels:
                qrels[query_id] = set()
            
            if qrel.relevance > 0:
                qrels[query_id].add(doc_id)
        
        self._qrels_cache = qrels
        return qrels
    
    def get_document_by_id(self, doc_id: str) -> Optional[str]:
        if self._documents_cache is None:
            self.load_documents()
        
        try:
            doc = self.dataset.docs_store().get(doc_id)
            return doc.text if doc else None
        except:
            return None
    
    def get_query_by_id(self, query_id: str) -> Optional[str]:
        if self._queries_cache is None:
            self.load_queries()
        
        return self._queries_cache.get(query_id)
    
    def get_relevant_docs(self, query_id: str) -> Set[str]:
        if self._qrels_cache is None:
            self.load_qrels()
        
        return self._qrels_cache.get(query_id, set())
    
    def filter_queries_with_relevant_docs(
        self, 
        queries: Dict[str, str], 
        qrels: Dict[str, Set[str]], 
        doc_index: Dict[str, int],
        min_relevant: int = 1
    ) -> Dict[str, str]:
        filtered = {}
        for query_id, query_text in queries.items():
            relevant_doc_ids = qrels.get(query_id, set())
            relevant_in_corpus = sum(1 for doc_id in relevant_doc_ids if doc_id in doc_index)
            if relevant_in_corpus >= min_relevant:
                filtered[query_id] = query_text
        return filtered

