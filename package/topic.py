from os.path import exists
import os
import nltk
from gensim.models import LdaMulticore
from gensim.corpora import dictionary
import logging

from utils import extractTokensWithID

class Topic():
    def __init__(self, nodes_file, items_file, number_topics):

        self.number_nodes = 0
        self.number_items = 0
        self._mapping = {} # mapping user id or item asin to bow
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
            separtedNum = len(nodes)
            nodes.extend(items)

            self._id2word = dictionary.Dictionary(nodes)
            self._corpus= [self._id2word.doc2bow(text) for text in nodes]

        self._stopwords_path = "./nltk_data/corpora/stopwords/"
        if not exists(self._stopwords_path):
            nltk.download("stopwords", download_dir=self._stopwords_path)
        logging.info("Download stopwords done.")

        node_ids, node_docs = _prepare(nodes_file)
        item_ids, item_docs = _prepare(items_file)
        self.number_nodes = len(node_docs)
        self.number_items = len(item_docs)

        logging.info("Prepare to construct corpora")
        _constructCorpus(node_docs, item_docs)
        logging.info("Construct LDA Model.")
        self._model = LdaMulticore(corpus=self._corpus, num_topics=number_topics, id2word=self._id2word)

        ids = node_ids
        ids.extend(item_ids)

        for i in range(len(ids)):
            self._mapping[ids[i]] = self._corpus[i]

    def __getitem__(self, id):
            return [pair[1] for pair in self._model[self._mapping[id]]]

    def save(self, path = "D:\\論文實驗\\data\\topic\\"):
        self._model.save(path + "topic")

    def load(self, path = "D:\\論文實驗\\data\\topic\\"):
        self._model.load(path + "topic")