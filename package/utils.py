import re

def preprocessingText(text):
    '''
        把標點符號和'\n'替換成空白，並且全部變為小寫
    '''
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text).lower()
    return text.replace('\n', ' ')

def extractTokensWithID(name, path_dataset):
    '''
        針對預處理後的不同資料集來源, 實作不同提取token的方法, 並且回傳物品或使用者id和對應的token
        Args:
            name (str): name of the dataset
            path_dataset (str): path of the dataset
        Returns:
            list of ids, list of tokens
    '''
    def extractAmazon(row):
        '''
            Order of the attrs of the item are asin, also_view, also_buy, category and price.
        '''
        return row[0], row[3].split(" ") # Third attr is category

    def extractDBLP(row):
        return row[0], row[1:]
    
    extractFunc = None
    if name.lower() == "amazon":
        extractFunc = extractAmazon
    elif name.lower() == "dblp":
        extractFunc = extractDBLP

    if not extractFunc:
        raise ValueError("Shoud choose one of the dataset for extracting tokens")
    
    list_tokens = []
    ids = []
    with open(path_dataset, "r", encoding="utf8") as f:
        for line in f:
            values = line.split(",")
            id,tokens = extractFunc(values)
            list_tokens.append(tokens)
            ids.append(id)
    return ids, list_tokens