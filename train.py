"""
author: Prabhu

"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.Performence_metrics import *
from src.Character_level_cnn import characterlevel

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level CNN for text classification""")
    parser.add_argument("-a", "--alphabet", type =str,
                        default= """abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")

    parser.add_argument("-m", "--max_length", type=int, default= 1014)
    parser.add_argument("-p", "--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default= 120)
    parser.add_argument("-n","--num_epochs", type= int, default= 10)
    parser.add_argument("-l", "--lr", type= float, choices=[0.01,0.001], default= 0.01, help=" recommended for sgd 0.01 and for adam 0.001")

    args = parser.parse_args()
    return args


