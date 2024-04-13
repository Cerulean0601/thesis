import networkx as nx
import queue
import random
import logging
import numpy as np
from copy import deepcopy

from package.topic import TopicModel
from package.utils import dot

class SN_Graph(nx.DiGraph):
    '''
        Note: 邊的權重若預設為1/(v_indegree)，則(v,u)和(u,v)的權重並不相同，因此用有向圖替代無向圖

        Param:
            located (bool): Whether the orignal social network is direct. Default is True.

        Attribute of Node:
            desired_set(string, Itemset)
            adopted_set(string, Itemset)
            adopted_records (list): It is a list of [Itemset, Coupon, int]. If users adopt items without coupons, the value is None.
            Third variable is the traded amount.

        Attribute of Edge:
            is_tested(bool):
            weight(float): 1/in_degree(u)
    '''
    def __init__(self, node_topic:TopicModel|dict = None, located = True) -> None:
        super().__init__()
        self.located = located # 原圖是否為無向圖，若為無向圖則重定向為有向圖
        self.topic = node_topic

    def __add__(self, another_G):
        '''
            It is a composed operation.If a edge or a node exists in both of the graphs, the second one will overwrite the first.
            The addtion of the two graphs can be used before the diffusion.
        '''
        if self.located != another_G.located:
            raise ValueError("Both of the graphs should be located, or neither.")
        
        compose = nx.compose(self, another_G)
        compose.located = self.located
        
        compose._initAllEdges()

        return compose

    def getAllTopics(self):
        return self.topic
    
    @staticmethod
    def construct(edges_file, node_topic:TopicModel|dict, located=True) -> None:
        '''
          從edge的資料檔案建立點, 邊, 權重

          Args:
            edges_file (string): 檔案路徑
            topic (Topic)
        '''
        graph = SN_Graph(node_topic, located=located)
        print("Constructing graph...")

        with open(edges_file, "r", encoding="utf8") as f:
            print("Connecting the edges...", end="")

            for line in f:
                src, det = line.split(",")
                det = det if det[-1] != "\n" else det[:-1]

                if src in node_topic and det in node_topic:
                    graph.add_edge(src, det)
                    if located:
                        graph.add_edge(det, src)

        graph.initAttr()
        print("Done")
        return graph
        
    def _bfs_sampling(self, num_nodes:int = None, roots:list = [], threshold=10**(-7)):

        if len(list(self.nodes)) == 0:
            raise Exception("The number of nodes in the original graph is zero.")

        def max_degree(out_degree):
            pair = (None, 0)
            for node, degree in list(out_degree):
                if pair[1] <= degree:
                    pair = (node, degree)
            return pair[0]

        if len(roots) == 0:
            roots.append(max_degree(self.out_degree))

        subgraph = SN_Graph(self.topic, self.convertDirected())
        q = queue.Queue()
        for node in roots:
            q.put(node)

        # bfs
        while not q.empty():
            if num_nodes != None and len(subgraph) <= num_nodes:
                break

            node = q.get()
            for out_neighbor, attr in self.adj[node].items():
                re_weight = 0.5*dot(self.nodes[node]["topic"], self.nodes[out_neighbor]["topic"]) + 0.5*attr["weight"]

                # if new weight is greater than threshold, then add the out_neighbor
                if re_weight > threshold and "is_sample" not in attr:
                    self.edges[node, out_neighbor]["is_sample"] = True
                    subgraph.add_edge(node, out_neighbor)
                    q.put(out_neighbor)

        q.task_done()
        subgraph.initAttr()
        return subgraph
      
    def sampling_subgraph(self, num_iter:int = 1, num_nodes:int = None, roots:list = [], threshold=10**(-7)):
        subgraph = SN_Graph(self.topic, located=self.convertDirected())

        for i in range(num_iter):
            population = deepcopy(self)
            subgraph += population._bfs_sampling(num_nodes, roots, threshold=threshold)
        return subgraph

    @staticmethod
    def transform(G, nodeTopic=None):
        '''
            Transform nx.Graph or nx.DiGraph to SN_Graph
        '''

        located = not G.is_directed()
        
        SN_G = SN_Graph(nodeTopic, located=located)
        for src, det in G.edges:
            src, det = src, det
            SN_G.add_edge(src, det)

        for node in G.nodes:
            if node not in SN_G.nodes:
                SN_G.add_node(node)

        SN_G.initAttr()

        return SN_G

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
                i = 0
                while i < len(l):
                    if l[i][1] <= ele[1]:
                        while i < len(l) and l[i][1] == ele[1] and l[i][0] <= ele[0]:
                            i += 1
                        break
                    i += 1
                l.insert(i, ele)

        topNodes = []

        for pair in self.out_degree:

            if len(topNodes) < k:
                insert(topNodes, pair)
            elif len(topNodes) == k and pair[1] > topNodes[-1][1]:
                insert(topNodes, pair)
                topNodes.pop(-1)

        return topNodes
    
    def convertDirected(self):
        return self.located

    def add_edge(self, src, det, **attr):
        if src not in self.nodes:
            self.add_node(src)
        
        if det not in self.nodes:
            self.add_node(det)
        
        if self.convertDirected():
            super().add_edge(det, src, **attr)
            self._initEdge(det, src, **attr)
            self._update_in_edge(src)

        super().add_edge(src, det, **attr)
        self._initEdge(src, det, **attr)
        self._update_in_edge(det)

    def add_edges_from(self, ebunch_to_add: zip|list, **attr):
        ebunch_to_add = list(ebunch_to_add)
        super().add_edges_from(ebunch_to_add, weight=0, is_tested=False, **attr)
        for edge in ebunch_to_add:
            src = edge[0]
            det = edge[1]
            self.add_node(src)
            self.add_node(det)
            if self.convertDirected():
                self._update_in_edge(src)
            
            self._update_in_edge(det)

    def add_node(self, node_for_adding, **attr):
        super().add_node(node_for_adding, **attr)
        self._initNode(node_for_adding, **attr)

    def _weightingEdge(self, node):
        return 1/self.in_degree(node)
    
    def _update_in_edge(self, node):
        for src, det in self.in_edges(node):
            self.edges[src, det]["weight"] = self._weightingEdge(det)

    def _initEdge(self, src, det, **attr):
        self.edges[src, det]["weight"] = self._weightingEdge(det)
        self.edges[src, det]["is_tested"] = False
        
        for key, value in attr.items():
            self.edges[src, det][key] = value

    def _initAllEdges(self):
        for src, det in list(self.edges):
            self._initEdge(src, det)
    
    def _initNode(self, id, **attr):
        self.nodes[id]["desired_set"] = None
        self.nodes[id]["adopted_set"] = None
        self.nodes[id]["adopted_records"] = list()
        if self.topic != None:
            self.nodes[id]["topic"] = self.topic[id]

        for key, value in attr.items():
            self.nodes[id][key] = value
            
    def _initAllNodes(self):
        for node in list(self.nodes):
            self._initNode(node)

    def initAttr(self):
        self._initAllEdges()
        self._initAllNodes()

    def resetAttr(self):
        for node_id in self:
            if self.nodes[node_id]["desired_set"] != None:
                self._initNode(node_id)
        
        for src, det in self.edges:
            self.edges[src, det]["is_tested"] = False
            
    @staticmethod
    def max_product_path(graph, seeds, cutoff=10**(-6)):

        # 取log是為了連乘=>log(a*b)=loga+logb
        # 取負數是要轉換為最短路徑問題
        nx.set_edge_attributes(graph, {(u, v): {"weight": -(np.log10(data["weight"])) if data["weight"] != 0 else 0} for u, v, data in graph.edges(data=True)})

        length, path = nx.multi_source_dijkstra(graph, seeds, weight="weight")

        for node, max_len in length.items():
            if node in seeds:
                length[node] = 1
            else:
                length[node] = np.power(10, -max_len)

        nx.set_edge_attributes(graph, {(u, v): {"weight":np.power(10, -max_len) if data["weight"] != 0 else 0} for u, v, data in graph.edges(data=True)})
        return length, path
    
    @staticmethod
    def compile_max_product_graph(graph, seeds):

        if not seeds:
            raise ValueError("The seed set is empty!")
        
        tree_graph = SN_Graph(node_topic=graph.getAllTopics(), located=graph.convertDirected())
        tree_graph._initAllEdges()
        length, path = SN_Graph.max_product_path(graph, seeds)

        for node, halfway in path.items():
            if len(halfway) == 1:
                tree_graph.add_node(node)

            for i in range(len(halfway)-1, 0, -1):
                # 從後面的節點開始連邊，如果邊已經連過了就跳出迴圈
                if not tree_graph.has_edge(halfway[i - 1], halfway[i]):
                    weight = length[halfway[i]] / length[halfway[i - 1]]
                    tree_graph.add_edge(halfway[i - 1], halfway[i], weight=weight)

        tree_graph._initAllNodes()

        return tree_graph, length