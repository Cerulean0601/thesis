import re
import networkx as nx
import numpy as np


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
            asin, also_view, also_buy, price = line.split(",")
            dataset[asin] = dict()
            dataset[asin]["also_view"] = also_view.split(" ")
            dataset[asin]["also_buy"] = also_buy.split(" ")
            dataset[asin]["price"] = float(price)

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

def calculate_inner_product_matrix(vectors):
    """
    計算向量之間的內積矩陣。
    
    參數:
    vectors - 長度為 N 的 list，每個元素為長度為 T 的 list。
    
    返回值:
    inner_product_matrix - N x N 的內積矩陣。
    """
    N = len(vectors)
    inner_product_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            inner_product_matrix[i][j] = np.dot(vectors[i], vectors[j])
            inner_product_matrix[j][i] = inner_product_matrix[i][j]
    return inner_product_matrix

def merge_super_node_vectors(v1, v2, count1, count2):
    """
    合併兩個向量，返回合併後的新向量。
    
    參數:
    v1, v2 - 要合併的兩個向量。
    count1, count2 - 兩個向量各自被合併的次數。
    
    返回值:
    new_vector - 合併後的新向量。
    new_count - 合併後的新向量的次數。
    """
    total_count = count1 + count2
    new_vector = [(x * count1 + y * count2) / total_count for x, y in zip(v1, v2)]
    return new_vector, total_count

def find_max_inner_product_indices(inner_product_matrix):
    """
    找到內積最大的兩個向量的索引。
    
    參數:
    inner_product_matrix - N x N 的內積矩陣。
    
    返回值:
    max_indices - 內積最大的兩個向量的索引。
    max_value - 內積的最大值。
    """
    max_value = -np.inf
    max_indices = (-1, -1)
    N = inner_product_matrix.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if inner_product_matrix[i][j] > max_value:
                max_value = inner_product_matrix[i][j]
                max_indices = (i, j)
    return max_indices, max_value

def update_inner_product_matrix(inner_product_matrix, vectors, new_vector, i, j):
    """
    更新內積矩陣，將合併後的新向量加入內積矩陣中，並刪除被合併的向量。
    
    參數:
    inner_product_matrix - N x N 的內積矩陣。
    vectors - 向量列表。
    new_vector - 合併後的新向量。
    i, j - 被合併的兩個向量的索引。
    
    返回值:
    updated_inner_product_matrix - 更新後的內積矩陣。
    """
    vectors.append(new_vector)
    
    # 刪除被合併的兩個向量
    vectors.pop(max(i, j))
    vectors.pop(min(i, j))
    
    # 移除內積矩陣中被合併向量的行和列
    inner_product_matrix = np.delete(inner_product_matrix, [i, j], axis=0)
    inner_product_matrix = np.delete(inner_product_matrix, [i, j], axis=1)
    
    # 計算新向量與其他向量的內積
    new_row = np.array([np.dot(new_vector, v) for v in vectors])

    # 新向量自身的內積
    new_inner_product_matrix = np.zeros((inner_product_matrix.shape[0] + 1, inner_product_matrix.shape[1] + 1))
    new_inner_product_matrix[:-1, :-1] = inner_product_matrix
    new_inner_product_matrix[-1] = new_row
    new_inner_product_matrix[:, -1] = new_row

    return new_inner_product_matrix

def aggregate_super_nodes(vectors, theta):
    """
    處理向量列表，直到所有內積小於閾值 theta。
    
    參數:
    vectors - 長度為 N 的 list，每個元素為長度為 T 的 list。
    theta - 內積的閾值。
    
    返回值:
    剩餘的向量列表。
    """
    vectors = [np.array(v) for v in vectors]
    counts = [1] * len(vectors)  # 每個向量的初始計數為 1
    inner_product_matrix = calculate_inner_product_matrix(vectors)
    merge_history = [{i} for i in range(len(vectors))]

    while True:
        (i, j), max_inner_product = find_max_inner_product_indices(inner_product_matrix)
        if max_inner_product < theta:
            break

        new_vector, new_count = merge_super_node_vectors(vectors[i], vectors[j], counts[i], counts[j])
        
        # Update the merge history
        merge_history.append(merge_history[i].union(merge_history[j]))
        merge_history.pop(max(i, j))
        merge_history.pop(min(i, j))

        counts.append(new_count)
        counts.pop(max(i, j))
        counts.pop(min(i, j))
        
        inner_product_matrix = update_inner_product_matrix(inner_product_matrix, vectors, new_vector, i, j)
    
    return [list(v) for v in vectors], merge_history
