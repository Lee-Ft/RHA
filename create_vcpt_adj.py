from __future__ import absolute_import, division, print_function

import os
import sys
import h5py
import pickle
import numpy as np
import torch
import copy
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from model.position_emb import build_graph, torch_broadcast_adj_matrix
from utils import get_vcpt_adj_mtx
from config import BaseOptions

if __name__ == "__main__":

    get_vcpt_adj_mtx()