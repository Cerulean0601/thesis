import networkx as nx
import queue
import random

from topic import TopicModel

class SN_Graph(nx.DiGraph):
    '''
        Note: 邊的權重若預設為1/(v_indegree)，則(v,u)和(u,v)的權重並不相同，因此用有向圖替代無向圖

        Param:
            isDirect (bool): Whether the orignal social network is direct. Default is False.

        Attribute of Node:
            desired_set(string)
            adopted_set(string)
        
        Attribute of Edge:
            is_tested(bool):
            weight(float): 1/in_degree(u)
    '''
    def __init__(self, isDirected=False) -> None:
        super().__init__()
        self.isDirected = isDirected

    def construct(self, edges_file, node_file, topic:TopicModel|dict) -> None:
        '''
          從edge的資料檔案建立點, 邊, 權重

          Args:
            edges_file (string): 檔案路徑
            nodes_file (string): 包含topic的節點資料路徑
            topic (Topic)
        '''
        with open(edges_file, "r", encoding="utf8") as f:
            for line in f:
                nodes = line.split(",")
                src = nodes[0]
                det = nodes[1][:-1]

                if src in topic and det in topic:
                    self.add_edge(src, det)

        self.initAttr(topic)

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
            插入排序選出前k個out degree最高的節點, 若 degree 相同則從 id 最小的開始

            Return:
                list : 節點id
        '''
        def insert(l: list, ele: tuple):
            if len(l) == 0:
                l.append(ele)
            else:
                for i in range(len(l)):
                    if l[i][1] <= ele[1]:
                        while i < len(l) and l[i][1] == ele[1] and l[i][0] <= ele[0]:
                            i += 1
                        l.insert(i, ele)
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
    
    def is_directed(self):
        return self.isDirected

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if not self.is_directed():
            super().add_edge(v_of_edge, u_of_edge, **attr)

        super().add_edge(u_of_edge, v_of_edge, **attr)

    def initAttr(self, topic):

        def _initEdgeAttr():
            for src, det in list(self.edges):
                self.edges[src, det]["weight"] = 1/self.in_degree(det)
                self.edges[src, det]["is_tested"] = False

        def _initNodeAttr(topic) -> bool:
            for node in list(self.nodes):
                self.nodes[node]["desired_set"] = None
                self.nodes[node]["adopted_set"] = None
                self.nodes[node]["topic"] = topic[node]

        _initEdgeAttr()
        _initNodeAttr(topic)
