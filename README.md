# thesis
論文實驗

# TODO
- [V] Diffusion model 的 load method 需改成 static method

# If Available
- [ ] Coupons 改為 flyweight pattern, 並且修改 save 和 load method

# Seed
- 高單價的商品優先分配給 out-degree 數較高的節點
- 取 k 個節點作為 seed，k等於min(商品數量, 節點數量)

# Note
- The structure of the directory should be "./dataset_name/raw_data"
- 演算法評估每個群的主商品及附加品時，可能會出現None的情況，目前先暫時刪除key=None的值。

# DBLP
## File
- dblp.json: The file is dblp dataset which had been preprocessed.
- dblpv13.json: Raw source dataset
- edges: The row "u v" is the edge from u to v.
- nodes: All of nodes id in DBLP
- sample{N}nodes_with_tokens.csv: Sampling N of the nodes with their own tokens
- ~~topic_items.csv: topic of the papers~~ (deprecated)
- token_nodes.csv: tokens of the nodes
    - format: id, token1, token2, ...

# Amazon
- meta_{category}.json: Raw source dataset for {category}
- preprocessed_{category}.json: Each of the transactions contain the properties which keep order, including asin, also_view, also_buy, category and price. 