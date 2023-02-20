from os.path import exists
import os
import nltk
from gensim.models import LdaMulticore
from gensim.corpora import dictionary
import logging
from random import random

import utils

class TopicModel():
    def __init__(self, number_topics, nodesTopic = {}, itemsTopic = {}):
        
        self._mappingNode = nodesTopic # mapping user id or item asin to bow
        self._mappingItem = itemsTopic
        self.number_topics = number_topics

    def construct(self, nodes_file, items_file):
        def _prepare(file):
            
            path = file.split(os.sep)
            path = [directory.lower() for directory in path]
            datasetName = path[-2]

            return utils.extractTokensWithID(datasetName, file)
        
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
        self._model = LdaMulticore(corpus=self._corpus, num_topics=self.number_topics, id2word=self._id2word)

        nodes_size = len(self._nodes_id)
        for i in range(nodes_size):
            bow = self._corpus[i]
            self._mappingNode[self._nodes_id[i]] = [pair[1] for pair in self._model[bow]]
        for j in range(len(self._items_id)):
            bow = self._corpus[nodes_size + j]
            self._mappingItem[self._items_id[j]] = [pair[1] for pair in self._model[bow]]

    def __getitem__(self, id):
        return self._mappingNode[id] if id in self._mappingNode else self._mappingItem[id]
    
    def getItemsTopic(self) -> dict:
        return self._mappingItem
    
    def getNodesTopic(self) -> dict:
        return self._mappingNode
    
    def randomTopic(self) -> list:
        topic = [random() for t in range(self.number_topics)]
        norm = sum(topic)
        normTopic = [t/norm for t in topic]
        
        return normTopic 


    def save(self, path = "D:\\論文實驗\\data\\topic\\"):
        def saveToFile(filename, _mapping):
            with open(filename, "w" ,encoding="utf8") as f:
                output = ""
                for numbering, topic in _mapping.items():
                    output += "{0},{1}\n".format(numbering, " ".join([str(t) for t in topic]))
        
                f.write(output)

        saveToFile(path + "topic" + str(self.number_topics) + "_items.csv", self._mappingItem)
        saveToFile(path + "topic" + str(self.number_topics) + "_users.csv", self._mappingNode)

    @staticmethod
    def load(number_topics, path = "D:\\論文實驗\\data\\topic\\") ->dict:
        '''
            Reload the topic vetors that is the output of LDA model which had been trained.
        '''
        def loadFromFile(filename):
            _mapping = dict()

            with open(filename, "r") as f:
                for line in f:
                    id, topic = line.split(",")
                    if id in _mapping:
                        raise ValueError("{0} of ids is duplicated.".format(id))
                    
                    topic = topic.split(" ")
                    _mapping[id] = [float(t) for t in topic]
        
            return _mapping

        return TopicModel(number_topics,  
                loadFromFile(path + "topic" + str(number_topics) + "_users.csv"),
                loadFromFile(path + "topic" + str(number_topics) + "_items.csv"))
        

        