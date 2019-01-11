#coding:utf-8
import argparse
import json
import sys
import os
import logging

from datautil import DataUtil
from model import *


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('KGModel Parser')
    parser.add_argument('--data_dir', type=str, default="data/FB15k-237/", nargs="?",
                        help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')

    parser.add_argument('--algorithm', type=str, default="HypER", nargs="?",
                        help='Which algorithm to use: HypER, ConvE, DistMult, or ComplEx')
    parser.add_argument('--mode', type=str, default="train", help='[train|demo]')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--reverse', type=bool, default=True, help='batch size')

    parser.add_argument('--optim_type', type=str, default="adam", help='optimizer')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')

    parser.add_argument('--ndim', type=int, default=200, help='hidden size')

    parser.add_argument('--entity_num', type=int, default=0, help='num of entities')
    parser.add_argument('--relation_num', type=int, default=0, help='num of relation')

    parser.add_argument('--epochs', type=int, default=500, help='num of relation')

    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--in_channels', type=int, default=1, help='in_channels')
    parser.add_argument('--out_channels', type=int, default=32, help='out_channels')
    parser.add_argument('--filt_h', type=int, default=1, help='filter_h')
    parser.add_argument('--filt_w', type=int, default=9, help='filter_w')

    return parser.parse_args()


args = parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    datautil = DataUtil(args)
    args.entity_num = len(datautil.entities)
    args.relation_num = len(datautil.relations)
    model = ConvE(args)
    datautil.train(model)
