import os
import sys
import re
import json
import pickle

from tqdm import tqdm
import h5py
import nltk
import numpy as np
import jsonlines
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from torchsummary import summary
from torch import optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# svm
x2 = [68.1, 67.4, 67.4]
y2 = [72.0, 71.6, 71.6]
# svm
x3 = [65.1, 62.6, 62.7]
y3 = [64.7, 62.9, 63.1]

x4 = [70.4, 70.1, 70.0]
y4 = [73.0, 72.7, 72.7]

x= [61.1, 60.6, 60.2]
y= [66.1, 65.3, 64.9]



x1 = np.array(x)
y1 = np.array(y)
print('multi-unimodal', y1 - x1)
tmp = (y1 - x1) / (100 - x1) * 100
print('error rate reduction', np.round(tmp, 1))
# print('error rate reduction', tmp)
