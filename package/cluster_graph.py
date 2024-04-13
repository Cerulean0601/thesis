import networkx as nx
from package.social_graph import SN_Graph
from package.utils import at_least_one_probability, expected_value
from queue import SimpleQueue
import sys
from numpy import dot, sum
from numpy.linalg import norm
class ClusterGraph(nx.DiGraph):
    def __init__(self, graph: SN_Graph, seeds:list[str], theta:float, depth:int, **attr):
        super().__init__()
        self._original_g = graph
        self._seeds = seeds
        self._compile(theta, depth)
    def _compile(self, theta: float, depth: int):
        seeds_attr = [(seed, self._original_g.nodes[seed]) for seed in self._seeds] 
        self.add_nodes_from(seeds_attr)
        current_level = SimpleQueue()
        next_level = SimpleQueue()

        for seed, attr in self.nodes(data=True):
            attr["nodes"] = set([seed])
            current_level.put(seed)

        for d in range(depth):
            while not current_level.empty():
                predecessor = current_level.get()
                entity_nodes = self.nodes[predecessor]["nodes"]
                out_neighbors = set()

                for node in entity_nodes:
                    out_neighbors = out_neighbors.union(set(self._original_g.neighbors(node)))

                _cluster_neighbors = self._clustering(out_neighbors, theta)
                for cluster in _cluster_neighbors:
                    topic = sum([self._original_g.nodes[n]["topic"] for n in cluster], axis=0)
                    topic = [t/len(cluster) for t in topic]
                    node_name = ", ".join([str(n) for n in cluster])

                    if not self.has_node(node_name):
                        self.add_node(node_name, 
                                    desired_set = None, 
                                    adopted_set = None, 
                                    nodes = cluster, 
                                    topic = topic)
                    
                    if not self.has_edge(predecessor, node_name):
                        weight = self._weighting(predecessor, node_name)
                        self.add_edge(predecessor, node_name, weight=weight)
                        next_level.put(node_name)

            next_level, current_level = current_level, next_level
    def _clustering(self, nodes: list, theta: float) -> list[list]:
        sum_cluster_vectors = []
        cluster_nodes = []

        if theta < -1 or theta > 1:
            raise ValueError("Theta must be between [0,1]")
        
        for n in nodes:
            topic = self._original_g.nodes[n]["topic"]
            i = -1
            min_cosine_sim = sys.float_info.max

            # find the cluster which the node belongs to, that the cosine simlarity is minimum
            for j in range(len(sum_cluster_vectors)):
                avg_vector = [t/len(cluster_nodes[j]) for t in sum_cluster_vectors[j]]
                cosine_sim = dot(topic, avg_vector)/(norm(topic)*norm(avg_vector))
                if  cosine_sim < min_cosine_sim and cosine_sim >= theta:
                    min_cosine_sim = cosine_sim
                    i = j

            if i == -1:
                sum_cluster_vectors.append(topic)
                cluster_nodes.append(set([n]))
            else:
                cluster_nodes[i].add(n)
                sum_cluster_vectors[i] = [sum(t) for t in zip(sum_cluster_vectors[i], topic)]
        
        return cluster_nodes
    
    def _weighting(self, parent, children) -> float:
        '''
        若子節點在上一層有多個父節點，則計算至少被一個節點影響成功的機率
        '''
        probabilities = []
        parent_nodes = self.nodes[parent]["nodes"]
        for child in self.nodes[children]["nodes"]:
            predecessors = set(self._original_g.predecessors(child)).intersection(parent_nodes)
            if(len(predecessors) > 1):
                pro = [self._original_g.edges[pre, child]["weight"] for pre in predecessors]
                weight = min(at_least_one_probability(pro), 1)
            elif(len(predecessors) == 1):
                weight = self._original_g.edges[list(predecessors)[0], child]["weight"]
            probabilities.append(weight)

        return expected_value(probabilities)