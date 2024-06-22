import re
import networkx as nx

def preprocessingText(text): # pragma: no cover
    '''
        把標點符號和'\n'替換成空白，並且全部變為小寫
    '''
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text).lower()
    return text.replace('\n', ' ')

def extractTokensWithID(name, path_dataset): # pragma: no cover
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
        next(f)
        for line in f:
            values = line.split(",")
            id,tokens = extractFunc(values)
            list_tokens.append(tokens)
            ids.append(id)
    return ids, list_tokens

def getItemsPrice(filename) -> list: # pragma: no cover
    prices = dict()

    with open(filename, "r", encoding="utf8") as dataFile:
        next(dataFile)
        for line in dataFile:
            id, *context, price = line.split(",")
            prices[id] = float(price[:-1])
    return prices

def read_items(filename): # pragma: no cover
    dataset = dict()
    with open(filename) as file:
        next(file)
        for line in file:
            asin, also_view, also_buy, for_coupon, price = line.split(",")
            dataset[asin] = dict()
            dataset[asin]["also_view"] = also_view.split(" ")
            dataset[asin]["also_buy"] = also_buy.split(" ")
            dataset[asin]["price"] = float(price)
            dataset[asin]["for_coupon"] = for_coupon
    return dataset

def dot(a:list, b:list):
    if len(a) != len(b):
        raise ValueError("The length of two topics must match")
    
    return sum(i[0]*i[1] for i in zip(a, b))

def exactly_one_probability(probabilities:list) -> float:
    '''
    恰好被一個節點影響成功的機率(已棄用)
    '''
    n = len(probabilities)
    dp = [[0]*n for i in range(n)]

    for i in range(n):
        dp[i][i] = 1 - probabilities[i]
        for j in range(0, i):
            dp[j][i] = dp[j][i-1] * dp[i][i]

    expected_value = probabilities[0] * dp[1][n-1]
    for i in range(1, n-1):
        expected_value += dp[0][i-1]*probabilities[i]*dp[i+1][n-1]
        
    expected_value += dp[0][n-2]*probabilities[n-1]

    return expected_value

def at_least_one_probability(probabilities: list) -> float:
    pro = 1
    for p in probabilities:
        pro *= p

    return 1-pro

def exactly_n_nodes(probabilities:list) -> list:
    n = len(probabilities)
    temp = [0] * (n + 1)
    temp[0] = 1
    prev = []
    
    for i in range(1, n + 1):
        prev = temp[:]
        for j in range(i + 1):
            if j == 0:
                temp[j] = prev[j] * (1 - probabilities[i - 1])
            else:
                temp[j] = prev[j] * (1 - probabilities[i - 1]) + prev[j-1] * probabilities[i - 1] 

    # expectation = sum(j * temp[j] for j in range(n + 1))
    return temp
