from math import degrees
import networkx as nx
import queue
import random
from itertools import combinations

class SN_Graph(nx.DiGraph):
    '''
        Note: 邊的權重若預設為1/(v_indegree)，則(v,u)和(u,v)的權重並不相同，因此用有向圖替代無向圖

        Attribute of Node:
            desired_set
            adopted_set
        
        Attribute of Edge:
            is_tested
            weight: 1/in_degree(u)
    '''
    def __init__(self) -> None:
        super().__init__(self)

  
    def construct(self, dataset) -> None:
        '''
          從edge的資料檔案建立點, 邊, 權重

          Args:
            dataset(string): 檔案路徑
        '''
    
        with open(dataset, "r") as f:
            for line in f:
                nodes = line.split(",")
                src = nodes[0]
                det = nodes[1] if nodes[1][-1] != "\n" else nodes[1][:-1]
                self.add_edge(src, det)
                self.edges[src, det]["is_tested"] = False
                self.nodes[src]["desired_set"] = None
                self.nodes[det]["adopted_set"] = None
                
        for src, det in list(self.edges):
            self.edges[src, det]["weight"] = 1/self.in_degree(det)

    def _bfs_sampling(self, k_nodes):

        if len(list(self.nodes)) == 0:
            raise Exception("The number of nodes in the original graph is zero.")

        def max_degree(self):
            pair = (None, 0)
            for node, degree in list(self.out_degree):
                if pair[1] <= degree:
                    pair = (node, degree)
            return pair[0]

        root = max_degree(self)

        subgraph = nx.DiGraph()
        q = queue.Queue()
        q.put(root)

        # bfs
        while not q.empty() and len(subgraph) <= k_nodes:
            node = q.get()
            for out_neighbor, attr in self.adj[node].items():
                if out_neighbor not in subgraph and random.random() < attr["weight"]:
                    subgraph.add_edge(
                    node, 
                    out_neighbor, 
                    weight = attr["weight"])
                    
                    subgraph.add_edge(
                      out_neighbor, 
                      node, 
                      weight = self.get_edge_data(out_neighbor, node, "weight"))
                    q.put(out_neighbor)

            q.task_done()

        return subgraph
      
    def sampling_subgraph(self, k_nodes, strategy="bfs") -> nx.DiGraph:
        return self._bfs_sampling(k_nodes)

    def top_k_nodes(self, k: int) -> list:
        '''
            插入排序選出前k個out degree最高的節點, 若 degree 相同則從 id 最低的開始

            Return:
                list : 節點id
        '''
        def insert(l: list, ele: tuple):
            if len(l) == 0:
                l.append(ele)
            else:
                for i in range(len(l)):
                    if l[i][1] < ele[1]:
                        l.insert(i+1, ele)
                        break
            
        topNodes = []
        nodes_degree = list(self.out_degree)

        for pair in nodes_degree:

            if len(topNodes) < k:
                insert(topNodes, pair)
            elif len(topNodes) == k and pair[1] > topNodes[-1][1]:
                topNodes.pop(-1)
                insert(topNodes, pair)
                
        return topNodes

                


