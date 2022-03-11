# -*- coding:utf-8 -*-
"""
@Time: 2022/03/11 20:34
@Author: KI
@File: args.py
@Motto: Hungry And Humble
"""
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--p', type=float, default=1, help='return parameter')
    parser.add_argument('--q', type=float, default=0.5, help='in-out parameter')
    parser.add_argument('--d', type=int, default=128, help='dimension')
    parser.add_argument('--r', type=int, default=10, help='walks per node')
    parser.add_argument('--l', type=int, default=80, help='walk length')
    parser.add_argument('--k', type=float, default=10, help='window size')

    args = parser.parse_args()

    return args
