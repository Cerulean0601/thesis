import networkx as nx
import queue
import random
from itertools import combinations

class SN_Graph(nx.DiGraph):
    '''
        Note: 邊的權重若預設為1/(v_indegree)，則(v,u)和(u,v)的權重並不相同，因此用有向圖替代無向圖
    '''
    def __init__(self) -> None:
        super().__init__(self)

  
    def construct(self, dataset) -> None:
        '''
          從edge的資料檔案建立點，邊，權重

          Args:
            dataset(string): 檔案路徑
        '''
    
        with open(dataset, "r") as f:
            for line in f:
                nodes = line.split(",")
                src = nodes[0]
                det = nodes[1] if nodes[1][-1] != "\n" else nodes[1][:-1]
                self.add_edge(src, det)

        for src, det in list(self.edges):
            self.edges[src,det]["weight"] = 1/self.in_degree(det)

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
  
    def random_topic(self, Z):
        '''
          為每個使用者隨機產生Z個Topic vector
        '''
        for node in self.nodes:
            rand_list = [random.random() for i in range(Z)]
      
      # Sum all elements, that is equal to one
            denominator = sum(rand_list)
            self.nodes[node]["topic"] = [i/denominator for i in rand_list]
      
    def sampling_subgraph(self, k_nodes, strategy="bfs") -> nx.DiGraph:
        return self._bfs_sampling(k_nodes)

