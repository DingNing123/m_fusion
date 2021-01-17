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

work_dir = '/m_fusion_data/'
DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"
dataset_json = json.load(open(DATA_PATH_JSON))
total_num = len(dataset_json.keys())
shows, speakers, sarcasms = set(), set(), set()
speakers_friends = set()
speakers_bbt =set() # The Big Bang Theory
for idx, ID in enumerate(dataset_json.keys()):
    show = dataset_json[ID]["show"]
    shows.add(show)
    speaker = dataset_json[ID]["speaker"]
    speakers.add(speaker)
    sarcasm = dataset_json[ID]["sarcasm"]
    sarcasms.add(sarcasm)
    if show == 'FRIENDS':
        speakers_friends.add(speaker)
    if show == 'BBT':
        speakers_bbt.add(speaker)

print('len speakers: ',len(speakers))

def get_total_num_show(show):
    num = []
    for idx, ID in enumerate(dataset_json.keys()):
        show_tmp = dataset_json[ID]["show"]
        if show_tmp == show:
            num.append(1)
        else:
            num.append(0)
    tmp = np.array(num)
    return np.sum(tmp)


show_dict = {}
for show in shows:
    tmp = get_total_num_show(show)
    show_dict[show] = round(tmp / total_num * 100 , 0)

print(show_dict)

def get_dict_show_sarcasm(sarcasm):

    num = []
    show_index = {'FRIENDS': 0, 'GOLDENGIRLS': 1, 'BBT': 2, 'SARCASMOHOLICS': 3}
    for idx, ID in enumerate(dataset_json.keys()):
        show_tmp = dataset_json[ID]["show"]
        sarcasm_tmp = dataset_json[ID]["sarcasm"]
        if sarcasm == sarcasm_tmp:
            num.append(show_index[show_tmp])

    result = {}
    for show in shows:
        x = np.array(num)
        x = sum(x == show_index[show])
        result[show] = x
    return result


for sarcasm in sarcasms:
    tmp = get_dict_show_sarcasm(sarcasm)
    print('sarcastic ', sarcasm)
    tmp = sorted(tmp.items(), key=lambda d: d[1], reverse=True)
    print(tmp)

def get_dict_speaker_sarcasm(speaker):
    num = []
    for idx, ID in enumerate(dataset_json.keys()):
        sarcasm_tmp = dataset_json[ID]["sarcasm"]
        speaker_tmp = dataset_json[ID]["speaker"]
        if speaker_tmp == speaker:
            if sarcasm_tmp:
                num.append(1)
            else:
                num.append(0)
    result = {}
    tmp = np.array(num)
    result[0] = sum(tmp == 0)
    result[1] = sum(tmp == 1)
    return result

print('friends:')
for speaker in speakers_friends:
    print(speaker)
    tmp = get_dict_speaker_sarcasm(speaker)
    tmp = sorted(tmp.items(), key=lambda d: d[0], reverse=True)
    print(tmp)

print('bbt:')
for speaker in speakers_bbt:
    print(speaker)
    tmp = get_dict_speaker_sarcasm(speaker)
    tmp = sorted(tmp.items(), key=lambda d: d[0], reverse=True)
    print(tmp)

test_speakers = ['HOWARD', 'SHELDON']

def getSpeakerIndependent_ours():
    train_ind_SI, test_ind_SI = [], []
    for idx, ID in enumerate(dataset_json.keys()):
        speaker = dataset_json[ID]["speaker"]
        if speaker in test_speakers:
            test_ind_SI.append(idx)
        else:
            train_ind_SI.append(idx)

    train_index, test_index = train_ind_SI, test_ind_SI
    return np.array(train_index), np.array(test_index)

(train_index, test_index) = getSpeakerIndependent_ours()
print(train_index.shape)
print(test_index.shape)
print(len(train_index)+len(test_index))