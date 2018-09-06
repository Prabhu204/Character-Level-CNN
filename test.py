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
import torch.nn.functional as F

from src.Performence_metrics import *
from src.Character_level_cnn import characterlevel
from src.dataset import Dataset_

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Character level CNN for text classification""")
    parser.add_argument("-b", "--batch_size", type=int, default= 120)
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-d", "--dataset", type=str, default="Data/dbpedia_csv")
    parser.add_argument("-g", "--gpu", default=False)
    parser.add_argument("-s", "--save_path", type= str, default="")
    parser.add_argument("-r", "--save_result", type=str, default="Result")
    parser.add_argument("-i","--import_model", default='trained_model')
    args = parser.parse_args()
    return args


def eval(opt):

    testdata_set = Dataset_(opt.dataset+os.sep+'test.csv',opt.dataset+os.sep+'classes.txt', opt.max_length)
    testdata_generator = DataLoader(testdata_set, batch_size=opt.batch_size, shuffle= False, num_workers=0)
    if torch.cuda.is_available():
        model = torch.load(opt.import_model)
    else:
        model = torch.load(opt.import_model, map_location=lambda storage, loc: storage)



    model.eval()
    test_true_label = []
    test_pred_prob = []
    for batch in testdata_generator:
        _, n_true_label = batch
        if opt.gpu:
            batch = [Variable(record).cuda() for record in batch]
        else:
            batch = [Variable(record)for record in batch]
        t_data, _ = batch
        with torch.no_grad():
            t_pred_label = model(t_data)
        t_pred_label= F.softmax(t_pred_label)
        test_pred_prob.append(t_pred_label)
        test_true_label.extend(n_true_label)
    test_pred_prob = torch.cat(test_pred_prob, 0)
    test_pred_prob = test_pred_prob.cpu().data.numpy()
    test_true_label = np.array(test_true_label)

    test_metrics = get_metrics(test_true_label, test_pred_prob, list_metrics=['Accuracy', 'Loss', 'Confusion_matrix'])

    res_file = open(opt.save_result + os.sep + "test_output.txt", "w")
    res_file.write(
        "Test Result: \nLoss {} \nAccuracy:{} \nConfusion_matrix:\n{}".format(test_metrics['Loss'],
                                                                              test_metrics['Accuracy'],
                                                                              test_metrics['Confusion_matrix']
                                                                              )
    )
    print("Test Result: \nLoss {} \nAccuracy:{} \nConfusion_matrix:\n{}".format(test_metrics['Loss'],
                                                                              test_metrics['Accuracy'],
                                                                              test_metrics['Confusion_matrix']
                                                                              ))

if __name__=="__main__":
    opt = get_args()
    eval(opt)



