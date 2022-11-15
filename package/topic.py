from os.path import exists
import nltk
from gensim.models import LdaMulticore
from gensim.corpora import dictionary

class Topic():
    def __init__(self, nodes_file, items_file, number_topics):

        def _prepareWords(file):

            ids = []
            docs = []
            with open(file, encoding="utf8") as f:
                for line in f:
                    id, *tokens = line.split(",")
                    ids.append(id)
                    docs.append(tokens)

            return ids, docs
        
        def _constructCorpora(nodes: list[list[str]], items: list[list[str]]):
            '''
                從 tokens 建立語料庫

                Args: 
                    nodes (list[list[str]]]): a list of tokens of users
                    items (list[list[str]]): a list of tokens of items
            '''
            separtedNum = len(nodes)
            nodes.extend(items)

            id2word = dictionary.Dictionary(nodes)
            corpora = [id2word.doc2bow(text) for text in nodes]

            return id2word, corpora

        self._stopwords_path = "./nltk_data/corpora/stopwords/"
        if not exists(self._stopwords_path):
            nltk.download("stopwords", download_dir=self._stopwords_path)
        print("Download stopwords done.")

        node_ids, node_docs = _prepareWords(nodes_file)
        item_ids, item_docs = _prepareWords(items_file)
        print("Prepare to construct corpora")
        id2word, corpora = _constructCorpora(node_docs, item_docs)
        print("Construct LDA Model.")
        self._model = LdaMulticore(corpus=corpora, num_topics=number_topics, id2word=id2word)

if __name__ == "__main__":
    topic = Topic(nodes_file="./data/dblp/topic_nodes",
                  items_file="./data/dblp/topic_nodes",
                  number_topics=5)
    topic._model.print_topics()