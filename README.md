# Character-level-CNN


An implentation of ConvNets using Pytorch based on research paper [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
 
 The topic classification of the text is a natural language processing.ConvNets were used for extracting infromation 
 from raw signals, ranging from computer vision applications to speech recognition and others.
 Where as, text is considered as a kind of raw signal at character level and applying one-dimensional ConvNets to it.   

**Datasets**

Dbpedia and Amazon polarity datasets were used.

**Setup**

Gpu: Nvidia GTX 1050 4GB

**Runtime**

For 10 epochs: 14h

For 25 epochs: _In progress_

**Result**

_Coming soon_


**usage:** 

Implementation of Character level CNN for text classification
    
    [-a ALPHABET] 
   
    [-m MAX_LENGTH]
    
    [-p {sgd,adam}]
     
    [-b BATCH_SIZE]
     
    [-n NUM_EPOCHS]
     
    [-l {0.01,0.001}] 
        
    [-d DATASET]
      
    [-g GPU] 
     
    [-s SAVE_PATH]
    
    [-t MODEL_NAME] 
   
    [-r SAVE_RESULT] 
   
    [-rn RESULT_NAME]
    
    [-i IMPORT_MODEL]
    

**example usage:**

For training a model:

python train.py -d 'Path to dataset' -n 'num of epochs' -s 'Directory name to save trained model' -r 'Path to save result'
-rn 'name_of_the_result_file.txt'

For testing a model:

python test.py -b 'batch_size' -i 'path to saved trained model for evaluation'

