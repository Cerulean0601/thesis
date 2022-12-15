from os.path import exists
import os
import nltk
from gensim.models import LdaMulticore
from gensim.corpora import dictionary
import logging

from utils import extractTokensWithID

class TopicModel():
    def __init__(self, nodes_file, items_file, number_topics):
        
        self._mappingNode = {} # mapping user id or item asin to bow
        self._mappingItem = {}
        self.number_topics = number_topics

        def _prepare(file):
            
            path = file.split(os.sep)
            path = [directory.lower() for directory in path]
            datasetName = path[-2]

            return extractTokensWithID(datasetName, file)
        
        def _constructCorpus(nodes: list[list[str]], items: list[list[str]]):
            '''
                從 tokens 建立語料庫

                Args: 
                    nodes (list[list[str]]]): a list of tokens of users
                    items (list[list[str]]): a list of tokens of items
            '''
            nodes.extend(items)

            _id2word = dictionary.Dictionary(nodes)
            _corpus= [_id2word.doc2bow(text) for text in nodes]

            return _id2word, _corpus

        self._stopwords_path = "./nltk_data/corpora/stopwords/"
        if not exists(self._stopwords_path):
            nltk.download("stopwords", download_dir=self._stopwords_path)
        logging.info("Download stopwords done.")

        self._nodes_id, node_docs = _prepare(nodes_file)
        self._items_id, item_docs = _prepare(items_file)

        logging.info("Prepare to construct corpora")
        self._id2word, self._corpus = _constructCorpus(node_docs, item_docs)
        logging.info("Construct LDA Model.")
        self._model = LdaMulticore(corpus=self._corpus, num_topics=number_topics, id2word=self._id2word)

        nodes_size = len(self._nodes_id)
        for i in range(nodes_size):
            self._mappingNode[self._nodes_id[i]] = self._corpus[i]
        for j in range(len(self._items_id)):
            self._mappingItem[self._items_id[j]] = self._corpus[nodes_size + j]

    def __getitem__(self, id):
        _mapping = self._mappingNode if id in self._mappingNode else self._mappingItem
        return [pair[1] for pair in self._model[_mapping[id]]]
    
    def getItemsTopic(self) -> dict:
        return self._mappingItem
    
    def getNodesTopic(self) -> dict:
        return self._mappingNode
        
    def save(self, path = "D:\\論文實驗\\data\\topic\\"):
        self._model.save(path + "topic" + str(self.number_topics))

    @classmethod
    def load(cls, number_topics, path = "D:\\論文實驗\\data\\topic\\"):
        return LdaMulticore.load(path + "topic" + str(number_topics))