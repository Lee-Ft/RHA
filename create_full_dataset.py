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
from utils import concat_json
from config import BaseOptions

if __name__ == "__main__":

    json_list = ["/home/data/tvqa_plus_stage_features/tvqa_plus_train_preprocessed.json", "/home/data/tvqa_plus_stage_features/tvqa_plus_valid_preprocessed.json"]
    concat_json(json_list, "/home/data/tvqa_plus_stage_features/tvqa_plus_train_preprocessed_full.json")