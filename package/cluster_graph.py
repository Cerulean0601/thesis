from package.social_graph import SN_Graph
# from package.utils import at_least_one_probability, exactly_n_nodes
from package.utils import aggregate_super_nodes
from package.topic import TopicModel

import networkx as nx
from queue import SimpleQueue
import sys
from numpy import dot, sum
from numpy.linalg import norm
from collections.abc import Iterator
import heapq
from copy import deepcopy

class EdgeWeight:
    def weighting(self):
        raise NotImplementedError("This method is not implemented.")

class TwoHopWeight(EdgeWeight):
    '''
    References:
    Purohit, M., Prakash, B. A., Kang, C., Zhang, Y., & Subrahmanian, V. S. (2014, August). 
    Fast influence-based coarsening for large networks. 
    In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1296-1305).
    '''
    def __init__(self, graph) -> None:
        self._graph = graph

    def _weighting(self, src, det, src_neighbors:set, det_neighbors:set, superNode):
        
        update_attr = dict()
        beta_1 = self._graph.edges[src, det]["weight"] if self._graph.has_edge(src, det) else 0
        beta_2 = self._graph.edges[det, src]["weight"] if self._graph.has_edge(det, src) else 0

        for src_neighbor in src_neighbors:
            update_attr[src_neighbor, superNode] = dict()
            if src_neighbor in det_neighbors:
                update_attr[src_neighbor, superNode]["weight"] = ((1+beta_1)*self._graph.edges[src_neighbor, src] + (1+beta_2)*self._graph.edges[src_neighbor, det])/4
            else:
                update_attr[src_neighbor, superNode]["weight"] = (1+beta_1)*self._graph.edges[src_neighbor, src]/2
        
        for det_neighbor in det_neighbors:
            update_attr[det_neighbor, superNode] = dict()
            if det_neighbor not in src_neighbors:
                update_attr[det_neighbor, superNode] = (1+beta_2)*self._graph.edges[det_neighbor, det]/2
        
    def weighting(self, u, v):
        # u 的 neighbors必須排除 v ， v的也必須排除u
        pass
class ClusterGraph(SN_Graph):
    def __init__(self, cluster_topic:TopicModel|dict = None, located = True, **attr):
        '''
          Other Args:
            graph: (SN_Graph)
            seeds: (list[str])
            theta: (float)
            depth: (int)
        '''
        super().__init__(node_topic=cluster_topic, located=located)
        if "graph" in attr:
            self._original_g, self._seeds = attr["graph"], attr["seeds"]
            self._compile(attr["theta"], attr["depth"])

    def _compile(self, theta: float, depth: int):
        seeds_attr = [(str(seed), deepcopy(self._original_g.nodes[seed])) for seed in self._seeds] 
        self.add_nodes_from(seeds_attr)
        current_level = SimpleQueue()
        next_level = SimpleQueue()

        for seed, attr in self.nodes(data=True):
            attr["nodes"] = set([seed])
            current_level.put(seed)

        for d in range(depth+1):
            while not current_level.empty():
                predecessor = current_level.get()
                entity_nodes = self.nodes[predecessor]["nodes"]
                out_neighbors = set()

                for node in entity_nodes:
                    out_neighbors = out_neighbors.union(set(self._original_g.neighbors(node)))
                
                # print("depth {} , number of neighbors: {}".format(d, len(out_neighbors)))
                out_neighbors = list(out_neighbors)
                super_nodes_topic, clustered_nodes = self._clustering(out_neighbors, theta)
                
                for i in range(len(clustered_nodes)):
                    
                    node_name = ", ".join([str(n) for n in sorted(clustered_nodes[i])])

                    if not self.has_node(node_name):
                        self.add_node(node_name, 
                                    desired_set = None, 
                                    adopted_set = None, 
                                    nodes = clustered_nodes[i], 
                                    topic = super_nodes_topic[i])
                    
                    if not self.has_edge(predecessor, node_name):
                        # TODO
                        self.add_edge(predecessor, node_name, weight=1, is_tested=False, is_active=True)
                        next_level.put(node_name)

            next_level, current_level = current_level, next_level

    def _clustering(self, nodes: list, theta: float) -> list[list]:
        vectors = [self._original_g.nodes[n]["topic"] for n in nodes]
        super_nodes_topic, clusters = aggregate_super_nodes(vectors, theta)
        clustered_nodes = []
        for cluster in clusters:
            collection = [nodes[node_index] for node_index in cluster]
            clustered_nodes.append(collection)

        return super_nodes_topic, clustered_nodes
    # def _clustering(self, nodes: list, theta: float) -> list[list]:
    #     sum_cluster_vectors = []
    #     cluster_nodes = []

    #     if theta < -1 or theta > 1:
    #         raise ValueError("Theta must be between [0,1]")
        
    #     for n in nodes:
    #         topic = self._original_g.nodes[n]["topic"]
    #         i = -1
    #         max_cosine_sim = sys.float_info.min
    #         # find the cluster which the node belongs to, that the cosine simlarity is minimum
    #         for j in range(len(sum_cluster_vectors)):
    #             avg_vector = [t/len(cluster_nodes[j]) for t in sum_cluster_vectors[j]]
    #             cosine_sim = dot(topic, avg_vector)/(norm(topic)*norm(avg_vector))
    #             if  cosine_sim > max_cosine_sim and cosine_sim >= theta:
    #                 max_cosine_sim = cosine_sim
    #                 i = j

    #         if i == -1:
    #             sum_cluster_vectors.append(topic)
    #             cluster_nodes.append(set([n]))
    #         else:
    #             cluster_nodes[i].add(n)
    #             sum_cluster_vectors[i] = [sum(t) for t in zip(sum_cluster_vectors[i], topic)]
        
    #     return cluster_nodes
    
    # def _weighting(self, parent, children) -> float:
    #     '''
    #     若子節點在上一層有多個父節點，則計算至少被一個節點影響成功的機率
    #     '''
    #     probabilities = []
    #     parent_nodes = self.nodes[parent]["nodes"]
    #     for child in self.nodes[children]["nodes"]:
    #         predecessors = set(self._original_g.predecessors(child)).intersection(parent_nodes)
    #         if(len(predecessors) > 1):
    #             pro = [self._original_g.edges[pre, child]["weight"] for pre in predecessors]
    #             weight = min(at_least_one_probability(pro), 1)
    #         elif(len(predecessors) == 1):
    #             weight = self._original_g.edges[list(predecessors)[0], child]["weight"]
    #         probabilities.append(weight)
        
    #     prob_distribution = exactly_n_nodes(probabilities)
    #     max_prob = max(prob_distribution)

    #     return max_prob, prob_distribution.index(max_prob)

        # return sum(probabilities)/len(probabilities), len(child)
    
    def level_travesal(self, sources:list, depth) -> Iterator[list]:

        yield sources

        level = set(s for s in sources)

        d = 0
        while len(level) > 0 and d < depth:
            next_level = set()
            for cluster in level:
                next_level = next_level.union(set(self.neighbors(cluster)))
            
            level = next_level
            d += 1
            yield sorted(list(level))

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        return super().add_weighted_edges_from(ebunch_to_add, weight, **attr)