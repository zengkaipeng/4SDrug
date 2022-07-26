import torch
import torch.nn.functional as F
from eval.metrics import multi_label_metric, ddi_rate_score
import numpy as np
from tqdm import tqdm
import argparse
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test')
    parser.add_argument('--path', required=True, help='the path of model')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--device', default=0, type=int, help='the cuda id')
    parser.add_argument('--cpu', action='store_true', help='use cpu for infer')
    args = parser.parse_args()
    args.dataset = 'MIMIC3'
    