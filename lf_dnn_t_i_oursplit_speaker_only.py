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


def print_hi(name):
    print(f'Hi, {name}')
    DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"
    BERT_TARGET_EMBEDDINGS = work_dir + "data/bert-output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = work_dir + "data/bert-output-context.jsonl"
    INDICES_FILE = work_dir + "data/split_indices.p"
    AUDIO_PICKLE = work_dir + "data/audio_features.p"
    BATCH_SIZE = 32
    model_path = os.path.join(work_dir + 'saved', f'lfdnn-mustard-M.pth')
    model_name = 'lf_dnn'
    RESULT_FILE = work_dir + "output/t_i_oursplit_speaker_only/{}.json"

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print()

    def pickle_loader(filename):
        if sys.version_info[0] < 3:
            return pickle.load(open(filename, 'rb'))
        else:
            return pickle.load(open(filename, 'rb'), encoding="latin1")

    class MMDataset(Dataset):
        def __init__(self, text_feature, video_feature_mean,
                     audio_feature_mean, authors_feature,
                     context_text_feature, label_out):
            # print('MMDataset')
            self.vision = video_feature_mean
            self.text = text_feature
            self.audio = audio_feature_mean
            self.label = label_out
            self.speaker = authors_feature
            self.context_text = context_text_feature
            # print('self.label')

        def __len__(self):
            return len(self.label)

        def __getitem__(self, index):
            sample = {
                'text': torch.Tensor(self.text[index]),
                'vision': torch.Tensor(self.vision[index]),
                'audio': torch.Tensor(self.audio[index]),
                'speaker': torch.Tensor(self.speaker[index]),
                'context_text': torch.Tensor(self.context_text[index]),
                'labels': torch.Tensor(self.label[index]).type(torch.LongTensor)
            }
            return sample

    dataset_json = json.load(open(DATA_PATH_JSON))
    print(type(dataset_json), len(dataset_json))  # dict 690
    print(list(dataset_json.keys())[:2])
    tmp = list(dataset_json.keys())[:2]
    print(tmp[0])
    tmp = dataset_json[tmp[0]]
    print(tmp)

    # text
    text_bert_embeddings = []

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

    def loadContextBert(dataset):
        length = []
        for idx, ID in enumerate(dataset.keys()):
            length.append(len(dataset[ID]["context"]))
        with jsonlines.open(BERT_CONTEXT_EMBEDDINGS) as reader:
            context_utterance_embeddings = []
            # Visit each context utterance
            for obj in reader:
                CLS_TOKEN_INDEX = 0
                features = obj['features'][CLS_TOKEN_INDEX]
                bert_embedding_target = []
                for layer in [0, 1, 2, 3]:
                    bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
                bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                context_utterance_embeddings.append(np.copy(bert_embedding_target))

        # Checking whether total context features == total context sentences
        assert (len(context_utterance_embeddings) == sum(length))

        # Rearrange context features for each target utterance
        cumulative_length = [length[0]]
        cumulative_value = length[0]
        for val in length[1:]:
            cumulative_value += val
            cumulative_length.append(cumulative_value)
        assert (len(length) == len(cumulative_length))
        end_index = cumulative_length
        start_index = [0] + cumulative_length[:-1]
        final_context_bert_features = []
        for start, end in zip(start_index, end_index):
            local_features = []
            for idx in range(start, end):
                local_features.append(context_utterance_embeddings[idx])
            final_context_bert_features.append(local_features)

        return final_context_bert_features

    # context bert embeddings
    context_bert_embeddings = loadContextBert(dataset_json)

    # video
    video_features_file = h5py.File(work_dir + 'data/features/utterances_final/resnet_pool5.hdf5')
    # combined feature index
    # audio dict (283 12)   (283 11)
    audio_features = pickle_loader(AUDIO_PICKLE)

    TEXT_ID = 0
    VIDEO_ID = 1
    AUDIO_ID = 2
    SHOW_ID = 3
    SPEAKER_ID = 4
    CONTEXT_BERT_ID = 5

    # parse_data
    data_input, data_output = [], []

    def parseData():
        # data_input [(text,video)(text,video)]
        # text:768 vide0: frame:2048
        for idx, ID in enumerate(dataset_json.keys()):
            # print(idx, 'processing ... ', ID) 0 processing ...  1_60
            data_input.append(
                (text_bert_embeddings[idx],  # 0 TEXT_ID
                 video_features_file[ID][()],  # 1 VIDEO_ID
                 audio_features[ID],  # 2
                 dataset_json[ID]["show"],  # 3 SHOW_ID
                 dataset_json[ID]["speaker"],  # 4
                 context_bert_embeddings[idx]  # 5
                 ))
            data_output.append(int(dataset_json[ID]["sarcasm"]))

    parseData()

    print('close video_features_file')
    video_features_file.close()

    splits = 5
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    split_indices = [(train_index, test_index) for train_index, test_index in skf.split(data_input, data_output)]
    print('split_indices: ')
    # print(split_indices)
    print(split_indices[0][0].shape, split_indices[0][1].shape)
    print(len(split_indices))
    # (552,)(138, )
    # 5

    if not os.path.exists(INDICES_FILE):
        pickle.dump(split_indices, open(INDICES_FILE, 'wb'), protocol=2)

    split_indices = pickle_loader(INDICES_FILE)
    print('after pickle_loader: ')
    print(split_indices[0][0].shape, split_indices[0][1].shape)
    print(len(split_indices))

    def toOneHot(data, size=None):
        '''
        Returns one hot label version of data
        '''
        oneHotData = np.zeros((len(data), size))
        oneHotData[range(len(data)), data] = 1

        assert (np.array_equal(data, np.argmax(oneHotData, axis=1)))
        return oneHotData

    def get_data_loader(train_ind_SI, author_ind):
        dataLoader = None
        # (text,video,AUDIO)
        train_input = [data_input[ind] for ind in train_ind_SI]
        # [0 1 0 1 ]
        train_out = np.array([data_output[ind] for ind in train_ind_SI])
        # expand dim (n,)  (n,1) it may be useless for crossentropy
        train_out = np.expand_dims(train_out, axis=1)

        def getData(ID=None):
            return [instance[ID] for instance in train_input]

        # Text Feature
        train_text_feature = getData(TEXT_ID)
        # video Feature
        train_video_feature = getData(VIDEO_ID)
        train_video_feature_mean = np.array([np.mean(feature_vector, axis=0) for feature_vector in train_video_feature])
        # audio Feature
        audio = getData(AUDIO_ID)
        # (552, 283)
        train_audio_feature = np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])

        # SPEAKER
        authors = getData(SPEAKER_ID)
        UNK_AUTHOR_ID = author_ind["PERSON"]
        authors = [author_ind.get(author.strip(), UNK_AUTHOR_ID) for author in authors]
        authors_feature = toOneHot(authors, len(author_ind))  # 18 speaker

        # CONTEXT BERT
        utterances = getData(CONTEXT_BERT_ID)
        mean_features = []
        for utt in utterances:
            mean_features.append(np.mean(utt, axis=0))
        context_text_feature = np.array(mean_features)

        train_dataset = MMDataset(train_text_feature, train_video_feature_mean, train_audio_feature,
                                  authors_feature, context_text_feature, train_out)

        train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
        dataLoader = train_dataLoader

        return dataLoader

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
        def __init__(self, speakers_num):
            super(LF_DNN, self).__init__()
            # 18 may be changed
            self.speakers_num = speakers_num
            self.text_in, self.context_text_in, self.video_in, self.audio_in, self.speaker_in = 768, 768, 2048, 283, speakers_num
            self.text_hidden, self.context_text_hidden, self.video_hidden, self.audio_hidden, self.speaker_hidden = 128, 32, 128, 32, 8
            # self.text_out = 32

            self.post_fusion_dim1 = 256
            self.post_fusion_dim2 = 32
            self.post_fusion_dim = 32
            self.video_prob, self.text_prob, self.context_text_prob, self.audio_prob, self.speaker_prob, self.post_fusion_prob = (
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2)

            self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
            self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
            self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob)
            self.context_text_subnet = SubNet(self.context_text_in, self.context_text_hidden, self.context_text_prob)

            self.speaker_subnet = SubNet(self.speaker_in, self.speaker_hidden, self.speaker_prob)

            self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
            self.post_fusion_dropout1 = nn.Dropout(p=self.post_fusion_prob)
            self.post_fusion_dropout2 = nn.Dropout(p=self.post_fusion_prob)

            self.post_fusion_layer_1 = nn.Linear( self.text_hidden + self.text_in , self.post_fusion_dim1)
            self.post_fusion_layer_4 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)
            self.post_fusion_layer_5 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)
            # self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim)

            self.post_fusion_layer_6 = nn.Linear(self.post_fusion_dim1 + self.text_in ,
                                                 self.post_fusion_dim2)

            self.post_fusion_layer_7 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)

            self.post_fusion_layer_8 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)

            self.post_fusion_layer_9 = nn.Linear(
                self.post_fusion_dim2 + self.speakers_num,
                self.post_fusion_dim)

            self.post_fusion_layer_10 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
            self.post_fusion_layer_11 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)

            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 2)

        def forward(self, text_x, video_x, audio_x, speaker_x, context_text_x):
            video_h = self.video_subnet(video_x)
            audio_h = self.audio_subnet(audio_x)
            text_h = self.text_subnet(text_x)
            context_text_h = self.context_text_subnet(context_text_x)
            speaker_h = self.speaker_subnet(speaker_x)

            # 128+32+16 = 176
            fusion_h = torch.cat([text_h,text_x], dim=-1)

            x = self.post_fusion_dropout(fusion_h)
            # x = self.post_fusion_layer_1(x)
            x = F.relu(self.post_fusion_layer_1(x), inplace=True)
            x = F.relu(self.post_fusion_layer_4(x), inplace=True)
            x = F.relu(self.post_fusion_layer_5(x), inplace=True)
            # x = F.relu(self.post_fusion_layer_2(x), inplace=True)

            x = torch.cat([x, text_x], dim=-1)

            x = self.post_fusion_dropout1(x)
            x = F.relu(self.post_fusion_layer_6(x), inplace=True)
            x = F.relu(self.post_fusion_layer_7(x), inplace=True)
            x = F.relu(self.post_fusion_layer_8(x), inplace=True)

            x = torch.cat([x, speaker_x], dim=-1)

            x = self.post_fusion_dropout2(x)
            x = F.relu(self.post_fusion_layer_9(x), inplace=True)
            x = F.relu(self.post_fusion_layer_10(x), inplace=True)
            x = F.relu(self.post_fusion_layer_11(x), inplace=True)

            output = self.post_fusion_layer_3(x)
            return output

    def print_model():
        # model = SubNet(2048,128,0.2)
        # model1 = SubNet(768,32,0.2)
        model2 = LF_DNN(18)
        model2.to(device)
        # summary(model,(2048,))
        # summary(model1,(768,))
        summary(model2, [(768,), (2048,), (283,), (18,), (768,)])
        # summary(model, [(1, 16, 16), (1, 28, 28)])
        return

    print_model()

    learning_rate = 5e-4
    weight_decay = 0.0
    early_stop = 20

    def do_test(model2, dataLoader, mode="VAL"):
        criterion = nn.CrossEntropyLoss()

        model2.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_acc = 0.0
        with torch.no_grad():
            with tqdm(dataLoader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(device)
                    text = batch_data['text'].to(device)
                    context_text = batch_data['context_text'].to(device)
                    audio = batch_data['audio'].to(device)
                    speaker = batch_data['speaker'].to(device)

                    labels = batch_data['labels'].to(device)

                    outputs = model2(text, vision, audio, speaker, context_text)

                    loss = criterion(outputs, labels.squeeze())

                    eval_loss += loss.item()
                    eval_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()
                    # eval_acc += torch.sum(m(outputs) >= 0.5).item()

                    # y_pred.append(outputs.cpu())
                    # y_true.append(labels.cpu())
                    y_pred.append(outputs.argmax(1).cpu())
                    y_true.append(labels.squeeze().long().cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # print('len pred: ', pred.size(),true.size())
        # 138 138
        # print('pred -- : ')
        # print(pred)

        eval_loss = eval_loss / len(pred)
        eval_acc = eval_acc / len(pred)
        # print('len dataLoader:',len(dataLoader))  1
        print("%s-(%s) >> loss: %.4f acc: %.4f" % (mode, 'lf_dnn', eval_loss, eval_acc))

        return eval_acc, pred, true

    def do_train(model2, train_dataLoader, val_dataLoader):
        best_acc = 0
        epochs, best_epoch = 0, 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
        while True:
            epochs += 1
            y_pred, y_true = [], []
            model2.train()
            train_loss = 0.0
            train_acc = 0.0
            with tqdm(train_dataLoader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(device)
                    audio = batch_data['audio'].to(device)
                    text = batch_data['text'].to(device)
                    context_text = batch_data['context_text'].to(device)
                    speaker = batch_data['speaker'].to(device)

                    labels = batch_data['labels'].to(device)

                    optimizer.zero_grad()
                    # forward
                    outputs = model2(text, vision, audio, speaker, context_text)

                    loss = criterion(outputs, labels.squeeze())
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    train_loss += loss.item()

                    train_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()

                    y_pred.append(outputs.argmax(1).cpu())
                    y_true.append(labels.squeeze().long().cpu())

            pred, true = torch.cat(y_pred), torch.cat(y_true)

            train_loss = train_loss / len(pred)

            train_acc = train_acc / len(pred)

            print("TRAIN-(%s) (%d/%d)>> loss: %.4f train_acc: %.4f" % (
                'lf_dnn', epochs - best_epoch, epochs, train_loss, train_acc))

            val_acc, _, _ = do_test(model2, val_dataLoader, mode="VAL")
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epochs
                print(model_path)
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(model2.cpu().state_dict(), model_path)
                model2.to(device)

            # early stop
            if epochs - best_epoch >= early_stop:
                print(f'the best epochs:{best_epoch},the best acc:{best_acc}')
                tmp = {
                    'best_epoch': best_epoch,
                    'best_acc': best_acc
                }
                return tmp
                # break

    def get_author_ind(train_ind_SI):
        # (text,video,AUDIO)
        train_input = [data_input[ind] for ind in train_ind_SI]

        def getData(ID=None):
            return [instance[ID] for instance in train_input]

        authors = getData(SPEAKER_ID)
        author_list = set()
        author_list.add("PERSON")

        for author in authors:
            author = author.strip()
            if "PERSON" not in author:  # PERSON3 PERSON1 all --> PERSON haha
                author_list.add(author)

        author_ind = {author: ind for ind, author in enumerate(author_list)}
        return author_ind

    results = []

    def five_fold():
        def getSpeakerIndependent():
            train_ind_SI, test_ind_SI = [], []
            for ind, data in enumerate(data_input):
                if data[SHOW_ID] == "FRIENDS":
                    test_ind_SI.append(ind)
                else:
                    train_ind_SI.append(ind)
            train_index, test_index = train_ind_SI, test_ind_SI
            return np.array(train_index), np.array(test_index)

        def getSpeakerIndependent_ours():
            train_ind_SI, test_ind_SI = [], []
            test_speakers = ['HOWARD', 'SHELDON']
            for idx, ID in enumerate(dataset_json.keys()):
                speaker = dataset_json[ID]["speaker"]
                if speaker in test_speakers:
                    test_ind_SI.append(idx)
                else:
                    train_ind_SI.append(idx)

            train_index, test_index = train_ind_SI, test_ind_SI
            return np.array(train_index), np.array(test_index)

        for fold in range(5):
            print(fold, '-' * 50)
            # (train_index, test_index) = getSpeakerIndependent()
            (train_index, test_index) = getSpeakerIndependent_ours()

            train_ind_SI = train_index
            val_ind_SI = test_index
            test_ind_SI = test_index

            print(train_ind_SI.shape, val_ind_SI.shape, test_ind_SI.shape)

            author_ind = get_author_ind(train_ind_SI)
            speakers_num = len(author_ind)

            train_dataLoader = get_data_loader(train_ind_SI, author_ind)
            val_dataLoader = get_data_loader(val_ind_SI, author_ind)
            test_dataLoader = get_data_loader(test_ind_SI, author_ind)

            model2 = LF_DNN(speakers_num)

            model2.to(device)

            best_result = do_train(model2, train_dataLoader, val_dataLoader)

            print()
            print(f'load:{model_path}')
            model2.load_state_dict(torch.load(model_path))
            model2.to(device)
            # do test
            val_acc, y_pred, y_true = do_test(model2, test_dataLoader, mode="TEST")
            print('Test: ', val_acc)
            # print(pred,true)
            result_string = classification_report(y_true, y_pred, digits=3)
            print('confusion_matrix(y_true, y_pred)')
            print(confusion_matrix(y_true, y_pred))
            print(result_string)

            result_dict = classification_report(y_true, y_pred, digits=3, output_dict=True)
            result_dict['best_result'] = best_result

            results.append(result_dict)

        # Dumping result to output
        if not os.path.exists(os.path.dirname(RESULT_FILE)):
            os.makedirs(os.path.dirname(RESULT_FILE))

        with open(RESULT_FILE.format(model_name), 'w') as file:
            json.dump(results, file)
        print('dump results  into ', RESULT_FILE.format(model_name))

        return None

    five_fold()

    def printResult(model_name=None):
        results = json.load(open(RESULT_FILE.format(model_name), "rb"))
        weighted_precision, weighted_recall = [], []
        weighted_fscores = []
        print("#" * 20)
        for fold, result in enumerate(results):
            weighted_fscores.append(result["weighted avg"]["f1-score"])
            weighted_precision.append(result["weighted avg"]["precision"])
            weighted_recall.append(result["weighted avg"]["recall"])
            print("Fold {}:".format(fold + 1))
            print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(
                result["weighted avg"]["precision"],
                result["weighted avg"]["recall"],
                result["weighted avg"]["f1-score"]))
            print(f'best_epoch:{result["best_result"]["best_epoch"]}   best_acc:{result["best_result"]["best_acc"]}')
        print("#" * 20)
        print("Avg :")
        print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(
            np.mean(weighted_precision),
            np.mean(weighted_recall),
            np.mean(weighted_fscores)))

        tmp_dict = {'precision': np.mean(weighted_precision), 'recall': np.mean(weighted_recall),
                    'f1': np.mean(weighted_fscores)}

        return tmp_dict

    printResult(model_name=model_name)


if __name__ == '__main__':
    print_hi('lf_dnn')
