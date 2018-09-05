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
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")
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
        #         labels.append(label)
        #         print(texts)
        #  (or) go for pandas library super time saver
        df = pd.read_csv(file_path, names=['0', '1', '2'])
        df['added_str'] = df['1']+' '+df['2']  # merge 2 columns strings with a space in between them
        self.texts = df['added_str'].tolist()
        self.labels = df['0'].tolist()
        self.char_wise_max_length_text = char_wise_max_length_text
        self.length = len(self.labels)
        if classes_file_path:
            self.num_classes = sum(1 for _ in open(classes_file_path))  # giving 1 for each class
                                                                        # in the classes file and then adding together.

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_matrix[self.vocabulary.index(i)]
                         for i in list(raw_text.lower()) if i in self.vocabulary], dtype= np.float32)
        if len(data)> self.char_wise_max_length_text:
            data = data[:self.char_wise_max_length_text]

        elif 0 < len(data) < self.char_wise_max_length_text:
            data = np.concatenate((data, np.zeros((self.char_wise_max_length_text - len(data), len(self.vocabulary)), dtype= np.float32)))

        elif len(data) == 0:
            data = np.zeros((self.char_wise_max_length_text, len(self.vocabulary)), dtype= np.float32)

        label = self.labels[index]
        return data, label





