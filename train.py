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
from src.dataset import Dataset_

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level CNN for text classification""")
    parser.add_argument("-a", "--alphabet", type =str,
                        default= """abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")

    parser.add_argument("-m", "--max_length", type=int, default= 1014)
    parser.add_argument("-p", "--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default= 120)
    parser.add_argument("-n","--num_epochs", type= int, default= 10)
    parser.add_argument("-l", "--lr", type= float, choices=[0.01,0.001], default= 0.01, help=" recommended for sgd 0.01 and for adam 0.001")
    parser.add_argument("-d", "--dataset",type=str, default="")
    args = parser.parse_args()
    return args

# def import_dataset(input_data, output_data):
#     input_data = "Data/dbpedia_csv"
#     output_data= "Result/dbpedia_csv"


def train(opt):

    res_file = open('Result'+os.sep + opt.dataset+ os.sep + "result.txt", "w")
    res_file.write("Select model parameters: {}".format(vars(opt)))

    training_set = Dataset_(opt.dataset+ os.sep+"train.csv", opt.dataset+os.sep+"classes.txt",
                            opt.max_length)

    train_data_generator = DataLoader(training_set, shuffle= True, num_workers= 0, batch_size= opt.batch_size)

    model = characterlevel(input_length= opt.max_length, n_classes= training_set.num_classes,
                           input_dim= len(opt.alphabet), n_convolutional_filter= 256, n_fc_neurons= 1024)


    criterion = nn.CrossEntropyLoss

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr= opt.lr, momentum=0.9)
    elif opt.optimizer=="adam":
        optimizer= torch.optim.Adam(model.parameters(),lr= opt.lr)

    model.train()






