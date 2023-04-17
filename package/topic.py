from os.path import exists
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

    def construct(self, items_file, nodes_file=None):
        '''
            construct topic model from tokens through LDA model
        '''
        def _prepare(file):
            
            path = file.split("/")
            path = [directory.lower() for directory in path]
            datasetName = path[-2]

            return utils.extractTokensWithID(datasetName, file)
        
        def _constructCorpus(docs: list[list[str]]):
            '''
                從 tokens 建立語料庫

                Args: 
                    nodes (list[list[str]]]): a list of tokens of item and users, if nodes_file is not None
            '''

            _id2word = dictionary.Dictionary(docs)
            _corpus= [_id2word.doc2bow(text) for text in docs]

            return _id2word, _corpus

        self._stopwords_path = "./nltk_data/corpora/stopwords/"
        if not exists(self._stopwords_path):
            nltk.download("stopwords", download_dir=self._stopwords_path)
        logging.info("Download stopwords done.")

        self._items_id, item_docs = _prepare(items_file)
        if nodes_file != None:
            self._nodes_id, node_docs = _prepare(nodes_file)
            item_docs.extend(node_docs)
            docs = item_docs

        logging.info("Prepare to construct corpora")
        self._id2word, self._corpus = _constructCorpus(docs)
        logging.info("Construct LDA Model.")
        self._model = LdaMulticore(corpus=self._corpus, num_topics=self.number_topics, id2word=self._id2word)

        for j in range(len(self._items_id)):
            bow = self._corpus[j]
            self._mappingItem[self._items_id[j]] = [pair[1] for pair in self._model.get_document_topics(bow, 0)]

        if nodes_file != None:
            bias = len(self._items_id)
            for i in range(bias, len(self._corpus)):
                bow = self._corpus[i]
                self._mappingNode[self._nodes_id[i]] = [pair[1] for pair in self._model.get_document_topics(bow, 0)]

    def read_topics(self, node_file=None, items_file=None):
        '''
            read topics which have been generated in file. The format of each line is id, topic_1
            topic_2, topic_3...
        '''
        if node_file!= None:
            with open(node_file, "r") as f:
                for line in f:
                    line = line.split(",")
                    id, topic = line[0], line[1]
                    topic = [float(t) for t in topic.split(" ")]
                    self._mappingNode[id] = topic

        if items_file!= None:
            with open(items_file, "r") as f:
                for line in f:
                    line = line.split(",")
                    id, topic = line[0], line[1]
                    topic = [float(t) for t in topic.split(" ")]
                    self._mappingItem[id] = topic
                    
    def __contains__(self, id):
        return id in self._mappingItem or id in self._mappingNode
    
    def __getitem__(self, id):
        return self._mappingNode[id] if id in self._mappingNode else self._mappingItem[id]
    
    def setItemsTopic(self, topic:dict):
        self._mappingItem = topic

    def getItemsTopic(self) -> dict:
        return self._mappingItem
    
    def setItemsTopic(self, topic:dict):
        self._mappingItem = topic

    def getNodesTopic(self) -> dict:
        return self._mappingNode
    
    def randomTopic(self, nodes_id=None, items_id=None) -> list:
        def generate(numberTopics):
            topic = [random() for t in range(numberTopics)]
            norm = sum(topic)
            normTopic = [t/norm for t in topic]
            
            return normTopic 

        if not nodes_id and not items_id:
            raise ValueError("The arguments of id should be pass at least one, nodes or items.")
        
        if nodes_id:
            for id in nodes_id:
                self._mappingNode[id] = generate(self.number_topics)
        
        if items_id:
            for asin in items_id:
                self._mappingItem[asin] = generate(self.number_topics)
        

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
        


        