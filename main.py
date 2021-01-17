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
import torch.nn.functional as F
from torchsummary import summary


def print_hi(name):
    print(f'Hi, {name}')
    DATA_PATH_JSON = "./data/sarcasm_data_ex2.json"
    dataset_json = json.load(open(DATA_PATH_JSON))
    print(type(dataset_json), len(dataset_json))  # dict 690
    print(list(dataset_json.keys())[:2])
    tmp = list(dataset_json.keys())[:2]
    print(tmp[0])
    tmp = dataset_json[tmp[0]]
    print(tmp)

    # text
    text_bert_embeddings = []
    BERT_TARGET_EMBEDDINGS = "./data/bert-output_ex2.jsonl"
    with jsonlines.open(BERT_TARGET_EMBEDDINGS) as reader:
        print('opend bert : ', BERT_TARGET_EMBEDDINGS)
        for obj in reader:
            CLS_TOKEN_INDEX = 0
            features = obj['features'][CLS_TOKEN_INDEX]
            bert_embedding_target = []
            for layer in [0, 1, 2, 3]:
                bert_embedding_target.append(np.array(features["layers"][layer]["values"]))

            bert_embedding_target = np.mean(bert_embedding_target, axis=0)
            # print(bert_embedding_target.shape) 768
            text_bert_embeddings.append(np.copy(bert_embedding_target))
    print('np.array(text_bert_embeddings).shape bert 768 ')
    print(np.array(text_bert_embeddings).shape)  # 690 768

    # video
    video_features_file = h5py.File('data/features/utterances_final_ex2/resnet_pool5.hdf5')

    # parse_data
    data_input, data_output = [], []
    # data_input [(text,video)(text,video)]
    # text:768 vide0: frame:2048
    for idx, ID in enumerate(dataset_json.keys()):
        print(idx, 'processing ... ', ID)
        data_input.append(
            (text_bert_embeddings[idx],
             video_features_file[ID][()]
             ))
        data_output.append(int(dataset_json[ID]["sarcasm"]))

    print(np.array(data_input[0][0]).shape)  # (768,)
    print(np.array(data_input[0][1]).shape)  # (72, 2048)  or   (96, 2048) not the same
    print()
    print(np.array(data_output).shape)
    print(data_output)  # [1, 1]

    print('close video_features_file')
    video_features_file.close()
    # train_index,test_index
    split_indices = [([0, 1], [0, 1])]
    train_ind_SI = [0, 1]
    test_ind_SI = [0, 1]

    train_input = [data_input[ind] for ind in train_ind_SI]
    train_out = np.array([data_output[ind] for ind in train_ind_SI])
    train_out = np.expand_dims(train_out, axis=1)
    print('train_out.shape -- ')
    print(train_out.shape) #(2, 1)

    size = 2
    train_oneHotData = np.zeros((len(train_out), size))
    train_oneHotData[range(len(train_out)), train_out] = 1
    print('oneHotData')
    print(train_oneHotData)

    print('train_out')
    # print(train_out) [1,1]
    # print(len(train_input),len(train_out)) 2 2
    # print( len(train_input[0][0])) 768

    test_input = [data_input[ind] for ind in test_ind_SI]
    test_out = [data_output[ind] for ind in test_ind_SI]

    TEXT_ID = 0
    text_feature = np.array([instance[TEXT_ID] for instance in train_input])
    print('text_feature.shape')
    print(text_feature.shape)
    # tmp now is equal to text_bert_embeddings
    # a lot of useless work
    VIDEO_ID = 1
    video_feature = [instance[VIDEO_ID] for instance in train_input]
    # print(np.array(tmp2[0]).shape) (96, 2048)
    length = [len(video) for video in video_feature]
    # print(length) 96 72

    video_feature_mean = np.array([np.mean(feature_vector, axis=0) for feature_vector in video_feature])
    print('video_feature_mean.shape')
    print(video_feature_mean.shape)

    class MMDataset(Dataset):
        def __init__(self, text_feature, video_feature_mean, train_out):
            print('MMDataset')
            self.vision = video_feature_mean
            self.text = text_feature
            self.label = train_out
            print('self.label')
            print(self.label)
            print(torch.Tensor(self.label))

        def __len__(self):
            return len(self.label)

        def __getitem__(self, index):
            # print('index')
            # print(index)
            sample = {
                'text': torch.Tensor(self.text[index]),
                'vision': torch.Tensor(self.vision[index]),
                'labels': torch.Tensor(self.label[index])
            }

            return sample

    BATCH_SIZE = 2
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

    print('device')
    print(device)
    dataset = MMDataset(text_feature, video_feature_mean, train_out)
    print('train_out --------')
    print(train_out)
    dataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    print('type(dataLoader)')
    print(type(dataLoader))
    print('dataset.vision.shape')
    print(dataset.vision.shape, len(dataset))
    print(dataset[0].__class__)
    print(dataset[0].keys())
    print(dataset[0]['text'].shape)
    print(dataset[0]['vision'].shape)
    print(dataset[0]['labels'].shape)
    print(dataset[0]['labels'])

    class SubNet(nn.Module):
        def __init__(self, in_size, hidden_size, dropout):
            super(SubNet, self).__init__()
            self.norm = nn.BatchNorm1d(in_size)
            self.drop = nn.Dropout(p=dropout)
            self.linear_1 = nn.Linear(in_size, hidden_size)
            self.linear_2 = nn.Linear(hidden_size, hidden_size)
            self.linear_3 = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            normed = self.norm(x)
            dropped = self.drop(normed)
            y_1 = F.relu(self.linear_1(dropped))
            y_2 = F.relu(self.linear_2(y_1))
            y_3 = F.relu(self.linear_3(y_2))
            return y_3

    class LF_DNN(nn.Module):
        def __init__(self):
            super(LF_DNN, self).__init__()
            self.text_in, self.video_in = 768, 2048
            self.text_hidden, self.video_hidden = 128, 128
            self.text_out = 32
            self.post_fusion_dim = 32
            self.video_prob, self.text_prob, self.post_fusion_prob = (0.2, 0.2, 0.2)
            self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
            self.text_subnet = SubNet(self.text_in, self.text_out, self.text_prob)
            self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
            self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden, self.post_fusion_dim)
            self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        def forward(self, text_x, video_x):
            video_h = self.video_subnet(video_x)
            text_h = self.text_subnet(text_x)
            fusion_h = torch.cat([video_h, text_h], dim=-1)
            x = self.post_fusion_dropout(fusion_h)
            x = F.relu(self.post_fusion_layer_1(x), inplace=True)
            x = F.relu(self.post_fusion_layer_2(x), inplace=True)
            output = self.post_fusion_layer_3(x)
            return output

    # model = SubNet(2048,128,0.2)
    # model1 = SubNet(768,32,0.2)
    model2 = LF_DNN()
    model2.to(device)
    # summary(model,(2048,))
    # summary(model1,(768,))
    # summary(model2,(768,),(2048,))
    summary(model2, [(768,), (2048,)])
    # summary(model, [(1, 16, 16), (1, 28, 28)])
    learning_rate = 5e-4
    weight_decay = 0.0
    early_stop = 2
    from torch import optim
    # criterion = nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_acc = 0
    epochs, best_epoch = 0, 0

    def do_test(model2, dataLoader, mode="VAL"):
        model2.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_acc = 0.0
        with torch.no_grad():
            with tqdm(dataLoader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(device)
                    text = batch_data['text'].to(device)
                    labels = batch_data['labels'].to(device).view(-1, 1)
                    outputs = model2(text, vision)
                    # loss = criterion(outputs, labels)
                    loss = criterion(m(outputs), labels)
                    # loss = criterion(outputs, labels.squeeze().long())
                    eval_loss += loss.item()
                    # eval_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()
                    eval_acc += torch.sum(m(outputs) >= 0.5).item()

                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # print('len pred: ', len(pred)) len pred:  2

        eval_loss = eval_loss / len(pred)
        eval_acc = eval_acc / len(pred)
        # print('len dataLoader:',len(dataLoader))  1
        print("%s-(%s) >> val_loss: %.4f val_acc: %.4f" % (mode,'lf_dnn', eval_loss, eval_acc))

        return  eval_acc

    m = nn.Sigmoid()

    while True:
        epochs += 1
        y_pred, y_true = [], []
        model2.train()
        train_loss = 0.0
        train_acc = 0.0
        with tqdm(dataLoader) as td:
            for batch_data in td:
                vision = batch_data['vision'].to(device)
                text = batch_data['text'].to(device)
                labels = batch_data['labels'].to(device).view(-1, 1)
                # print('vision.shape')
                # print(vision.shape, text.shape, labels.shape)
                # print(labels)
                # clear gradient
                optimizer.zero_grad()
                # forward
                outputs = model2(text, vision)
                print('outputs.shape')
                print(outputs.shape,labels.shape)
                loss = criterion(m(outputs), labels)
                # backward
                loss.backward()
                # update
                optimizer.step()
                train_loss += loss.item()
                # print('before train_acc: ', train_acc) 0
                # print('argmax: ',outputs.argmax(1)) argmax:  tensor([1, 1])
                # print('labels long : ',torch.squeeze(labels.long()))
                # labels long :  tensor([1, 1])
                # train_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()
                train_acc += torch.sum(m(outputs) >= 0.5).item()
                print('train_acc: ---')
                print(train_acc)
                # print('after  train_acc: ', train_acc) 1 0.5 0

                # print('size:')
                # print(outputs.cpu().size()) torch.Size([2, 1])
                # print(labels.cpu().size()) torch.Size([2, 1])

                y_pred.append(outputs.cpu())
                y_true.append(labels.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # print('len pred: ', len(pred)) len pred:  2

        train_loss = train_loss / len(pred)
        train_acc = train_acc / len(pred)
        # print('len dataLoader:',len(dataLoader))  1
        print("TRAIN-(%s) (%d/%d)>> loss: %.4f train_acc: %.4f" % ('lf_dnn',
                                                             epochs - best_epoch, epochs, train_loss, train_acc))

        # print('pred.size true.size:')
        # print(pred.size() , true.size())  #torch.Size([2, 1]) torch.Size([2, 1])
        # default is dim 0,so it will be [6,1] [690,1]
        # print(pred, true)
        """
        tensor([[1.0882, 1.3037],
        [0.9390, 1.1894]], grad_fn=<CatBackward>) tensor([[1.],
        [1.]])
        """

        val_acc = do_test(model2, dataLoader, mode="VAL")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epochs
            model_path = os.path.join('saved', \
                                      f'lfdnn-mustard-M.pth')
            print(model_path)
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(model2.cpu().state_dict(), model_path)
            model2.to(device)

        # early stop
        if epochs - best_epoch >= early_stop:
            return



if __name__ == '__main__':
    print_hi('lf_dnn')
