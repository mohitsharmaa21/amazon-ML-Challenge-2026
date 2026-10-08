#!/usr/bin/python
import logging
from typing import List, Tuple

from entilink.canopy_index import CanopyIndex
from entilink.core import Enumerator
from entilink.index import Index

logger = logging.getLogger(__name__)

Doc = Tuple[str, ...]


class TfIdfIndex(Index):
    def __init__(self) -> None:
        self._index = CanopyIndex()
        self._doc_to_id = Enumerator(start=1)
        self._parseTerms = self._index.lexicon.parseTerms

    def index(self, doc: Doc) -> None:
        if doc not in self._doc_to_id:
            i = self._doc_to_id[doc]
            self._index.index_doc(i, doc)

    def unindex(self, doc) -> None:
        i = self._doc_to_id.pop(doc)
        self._index.unindex_doc(i)
        self.initSearch()

    def initSearch(self) -> None:
        self._index.initSearch()

    def search(self, doc: Doc, threshold: float = 0) -> List[int]:
        query_list = self._parseTerms(doc)

        if query_list:
            results = [
                center for score, center in self._index.apply(query_list, threshold)
            ]
        else:
            results = []

        return results
