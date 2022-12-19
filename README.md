# thesis
論文實驗

# TODO
- [ ] d
# Seed
- 高單價的商品優先分配給 out-degree 數較高的節點
- 取 k 個節點作為 seed，k等於min(商品數量, 節點數量)

# Note
- The structure of the directory should be "./dataset_name/raw_data"

# DBLP
## File
- dblp.json: The file is dblp dataset which had been preprocessed.
- dblpv13.json: Raw source dataset
- edges: The row "u v" is the edge from u to v.
- nodes: All of the node id in DBLP
- sample{N}topic_nodes.csv: Sampling N of the nodes with their own topics
- ~~topic_items.csv: topic of the papers~~ (deprecated)
- topic_nodes.csv: topic of the nodes
    - format: id, token1, token2, ...

# Amazon
- meta_{category}.json: Raw source dataset for {category}
- preprocessed_{category}.json: Each of the transactions contain the properties which keep order, including asin, also_view, also_buy, category and price. 