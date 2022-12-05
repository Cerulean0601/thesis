import re

def preprocessingText(text):
    '''
        把標點符號和'\n'替換成空白，並且全部變為小寫
    '''
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text).lower()
    return text.replace('\n', ' ')

def extractTokens(name, path_dataset):
    def extractAmazonTokens(row):
        '''
            Order of the attrs of the item are asin, also_view, also_buy, category and price.
        '''
        return row[3].split(" ") # Third attr is category

    def extractDBLPTokens(row):
        return row[1:]
    
    extractFunc = None
    if name.lower() == "amazon":
        extractFunc = extractAmazonTokens
    elif name.lower() == "dblp":
        extractFunc = extractDBLPTokens

    if not extractFunc:
        raise ValueError("Shoud choose one of the dataset for extracting tokens")
    
    list_tokens = []
    with open(path_dataset, "r") as f:
        for line in f:
            values = line.split(",")
            tokens = extractFunc(values)
            list_tokens.append(tokens)
    return list_tokens