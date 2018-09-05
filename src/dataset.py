"""
author: Prabhu

"""
import numpy as np
import sys
import csv
from torch.utils.data import Dataset

class Dataset_(Dataset):
    def __init__(self, file_path = None, classes_file_path = None, char_wise_max_length_text = 1014):
        self.file_path = file_path
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")
        # self.vocabulary = set(self.vocabulary)
        self.identity_matrix = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(file_path) as csv_file:
            r = csv.reader(csv_file, quotechar = '"')
            for idx, line in enumerate(r):
                text = ""
                for tx in line[1:]:
                    # print(tx)
                    text += tx
                    text += " "
                # print(text)
                label = int(line[0])-1
                # print(label)
                texts.append(text)
                labels.append(label)
                print(texts)
dt = Dataset_(file_path= 'Data/dbpedia_csv/test.csv')


