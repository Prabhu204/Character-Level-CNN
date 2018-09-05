"""
author: Prabhu

"""
import numpy as np
import sys
import csv
from torch.utils.data import Dataset
import pandas as pd

class Dataset_(Dataset):
    def __init__(self, file_path = None, classes_file_path = None, char_wise_max_length_text = 1014):
        self.file_path = file_path
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")
        # self.vocabulary = set(self.vocabulary)
        self.identity_matrix = np.identity(len(self.vocabulary))
        # texts, labels = [], []
        # with open(file_path) as csv_file:
        #     r = csv.reader(csv_file, quotechar = '"')
        #     for idx, line in enumerate(r):
        #         text = ""
        #         for tx in line[1:]:
        #             # print(tx)
        #             text += tx
        #             text += " "
        #         # print(text)
        #         label = int(line[0])-1
        #         # print(label)
        #         texts.append(text)
        #
        #         labels.append(label)
        #         print(texts)
        #  (or) go for pandas library super time saver
        df = pd.read_csv(file_path, names=['0', '1', '2'])
        df['added_str'] = df['1']+' '+df['2']  # merge 2 columns strings with a space in between them
        # df.to_csv('Data/new.csv', index= False) # if you want you can save it locally
        self.texts = df['added_str'].tolist()
        self.labels = df['0'].tolist()



if __name__=='__main__':
    Dataset_(file_path= 'Data/dbpedia_csv/test.csv')


