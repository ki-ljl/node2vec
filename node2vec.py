# -*- coding: utf-8 -*-
"""
@Time ： 2021/12/17 21:44
@Author ：KI
@File ：node2vec.py
@Motto：Hungry And Humble

"""
import numpy as np
import sys
sys.path.append('../')
import numpy.random as npr
from gensim.models import Word2Vec


class node2vec:
    def __init__(self, args, G):
        self.G = G
        self.args = args
        self.init_transition_prob()

    def init_transition_prob(self):
        """
        :return:Normalized transition probability matrix
        """
        g = self.G
        nodes_info, edges_info = {}, {}
        for node in g.nodes:
            nbs = sorted(g.neighbors(node))
            probs = [g[node][n]['weight'] for n in nbs]
            # Normalized
            norm = sum(probs)
            normalized_probs = [float(n) / norm for n in probs]
            nodes_info[node] = self.alias_setup(normalized_probs)

        for edge in g.edges:
            # directed graph
            if g.is_directed():
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1])
            # undirected graph
            else:
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1])
                edges_info[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.nodes_info = nodes_info
        self.edges_info = edges_info

    def get_alias_edge(self, t, v):
        """
        Get the alias edge setup lists for a given edge.
        """
        g = self.G
        unnormalized_probs = []
        for v_nbr in sorted(g.neighbors(v)):
            if v_nbr == t:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / self.args.p)
            elif g.has_edge(v_nbr, t):
                unnormalized_probs.append(g[v][v_nbr]['weight'])
            else:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / self.args.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    def alias_setup(self, probs):
        """
        :probs: probability
        :return: Alias and Prob
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []  #
        larger = []  #
        for kk, prob in enumerate(probs):
            q[kk] = K * prob  #
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.

        #
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large  #
            q[large] = q[large] - (1.0 - q[small])  #

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def alias_draw(self, J, q):
        """
        in: Prob and Alias
        out: sampling results
        """
        K = len(J)
        # Draw from the overall uniform mixture.
        kk = int(np.floor(npr.rand() * K))  # random

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if npr.rand() < q[kk]:  # compare
            return kk
        else:
            return J[kk]

    def node2vecWalk(self, u):
        g = self.G
        walk = [u]
        nodes_info, edges_info = self.nodes_info, self.edges_info
        while len(walk) < self.args.l:
            curr = walk[-1]
            v_curr = sorted(g.neighbors(curr))
            if len(v_curr) > 0:
                if len(walk) == 1:
                    # print(adj_info_nodes[curr])
                    # print(alias_draw(adj_info_nodes[curr][0], adj_info_nodes[curr][1]))
                    walk.append(v_curr[self.alias_draw(nodes_info[curr][0], nodes_info[curr][1])])
                else:
                    prev = walk[-2]
                    ne = v_curr[self.alias_draw(edges_info[(prev, curr)][0], edges_info[(prev, curr)][1])]
                    walk.append(ne)
            else:
                break

        return walk

    def learning_features(self):
        walks = []
        g = self.G
        nodes = list(g.nodes())
        for t in range(self.args.r):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vecWalk(node)
                walks.append(walk)
        # embedding
        walks = [list(map(str, walk)) for walk in walks]
        # print(walks[0])
        # print(walks[1])
        model = Word2Vec(sentences=walks, vector_size=self.args.d, window=self.args.k, min_count=0, sg=1, workers=3)
        f = model.wv
        res = [f[x] for x in nodes]
        return res


