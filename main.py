# -*- coding:utf-8 -*-
"""
@Time: 2022/03/11 20:39
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from node2vec import node2vec
import networkx as nx


def main():
    args = args_parser()
    G = nx.davis_southern_women_graph()
    # G = nx.karate_club_graph()
    nodes = list(G.nodes.data())
    # G = nx.karate_club_graph()
    G = nx.les_miserables_graph()
    # for u, v in G.edges:
    #     G.add_edge(u, v, weight=1)
    vec = node2vec(args, G)
    model = vec.learning_features()


if __name__ == '__main__':
    main()
