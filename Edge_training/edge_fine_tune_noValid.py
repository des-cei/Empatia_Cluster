"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 12:13 PM 2024/05/13   
@FileName: CNN_LSTM_approach1.py
@Software: PyCharm


Introduction of this File:

"""
import itertools
import json
import sys
import os
import time
from random import shuffle

sys.path.append("/home/junjiao/PycharmProjects/Empatia_cluster/")
from Feature_wise_Normalization.Data_extraction import Abnormal_detection
from Feature_wise_Normalization.Data_normalization import Normalize_data


from sklearn.metrics import f1_score

from Training.models.cnn_lstm import ParallelCNNLSTMModel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import argparse
from Training.models import *
from Training.models import cnn_lstm
from Feature_wise_Normalization.Create_feature_maps import get_normalization, generate_feature_maps
from Training.utils import progress_bar, get_clusters

def Normalize0to255(array):
    """
    Normalize the array
    """
    for index in range(array.shape[2]):
        # Remember !!! The data in same column is the same feature.
        for index_f in range(array.shape[1]):
            mx = np.nanmax(array[:, :, index][:, index_f])
            mn = np.nanmin(array[:, :, index][:, index_f])
            # Suppress/hide the warning
            np.seterr(invalid='ignore')
            array[:, :, index][:, index_f] = np.round((array[:, :, index][:, index_f] - mn) / (mx - mn) * 255)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array

def Normalize0to1(array):
    """
    Normalize the array
    """
    for index in range(array.shape[2]):
        # Remember !!! The data in same column is the same feature.
        for index_f in range(array.shape[1]):
            mx = np.nanmax(array[:, :, index][:, index_f])
            mn = np.nanmin(array[:, :, index][:, index_f])
            # Suppress/hide the warning
            np.seterr(invalid='ignore')
            array[:, :, index][:, index_f] = np.round((array[:, :, index][:, index_f] - mn) / (mx - mn) * 1)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array

def feature_map_trans(feature_maps):
    feature_maps_trans = np.zeros((feature_maps.shape[2], 1, feature_maps.shape[0], feature_maps.shape[1]))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, 0, :, :] = feature_map
    return feature_maps_trans

def get_train_test_fine(cluster_set):
    # Randomly split participants
    cluster_list = list(cluster_set)
    n_total = len(cluster_list)
    offset = round(n_total * 0.8)
    if n_total == 0 or offset < 1:
        return [], cluster_list
    # random split the train and test
    shuffle(cluster_list)
    train_set = cluster_list[:offset]
    test_set = cluster_list[offset:]
    return train_set, test_set


best_acc = 0  # best test accuracy
best_f1 = 0.0

def Generate_dataloader(data_file, log_file, Picked_user, featureMap_stratgy, loader_size=None):

    test_feature_all_dic = {}
    test_label_all_dic = {}
    with open(data_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    for participant in content.items():
        for video in participant[1].items():
            for episode in video[1].values():
                if 'ReportadaBinarizado' in episode.keys():
                    for feature_name in episode.keys():
                        # Judge if the feature from the participants from train or test
                        participant_modified = int(participant[0].replace('V', ''))
                        if participant_modified == Picked_user:
                            if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                                if feature_name not in test_feature_all_dic.keys():
                                    test_feature_all_dic[feature_name] = [episode[feature_name]]
                                else:
                                    test_feature_all_dic[feature_name].append(episode[feature_name])
                            elif ('ReportadaBinarizado' in feature_name):
                                if feature_name not in test_label_all_dic.keys():
                                    test_label_all_dic[feature_name] = [episode[feature_name]]
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'] = [participant[0]]
                                    # put the number of video to label
                                    test_label_all_dic['Video'] = [video[0]]
                                else:
                                    test_label_all_dic[feature_name].append(episode[feature_name])
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'].append(participant[0])
                                    # put the number of video to label
                                    test_label_all_dic['Video'].append(video[0])
    test_feature_name_list = []
    test_feature_array = np.zeros(1)
    for index, feature in enumerate(test_feature_all_dic.items()):
        if index == 0:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.array(feature[1])
        else:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.vstack((test_feature_array, np.array(feature[1])))
    test_label_array = np.array(test_label_all_dic['ReportadaBinarizado'])
    test_feature_array = test_feature_array.transpose()
    test_feature_array[np.isnan(test_feature_array)] = 0
    test_feature_array[np.isinf(test_feature_array)] = 0
    test_P_label = test_label_all_dic['Participant']
    test_V_label = test_label_all_dic['Video']

    test_filter_array = Abnormal_detection(test_feature_array)
    test_Nor_feature_array = Normalize_data(Normalization_log_file=log_file,
                                            feature_array=test_filter_array,
                                            If0to1=True)
    test_P_label = np.array([int(i.replace('V', '')) for i in test_P_label])
    test_V_label = np.array([float(i) for i in test_V_label])
    test_array_all = np.vstack((test_Nor_feature_array, test_label_array, test_P_label, test_V_label))
    test_array_all[np.isnan(test_array_all)] = 0
    test_array_all[np.isinf(test_array_all)] = 0

    test_feature_map, test_label = generate_feature_maps(test_array_all, strategy=featureMap_stratgy)
    if featureMap_stratgy == 'All_concat':
        test_feature_map1, test_label1 = generate_feature_maps(test_array_all, strategy='AllFromOne')
        test_feature_map2, test_label2 = generate_feature_maps(test_array_all, strategy='HalfAndHalf')
        test_feature_map3, test_label3 = generate_feature_maps(test_array_all, strategy='HalfAndRandom')

        test_feature_map = np.dstack((test_feature_map1, test_feature_map2))
        test_feature_map = np.dstack((test_feature_map, test_feature_map3))
        test_label = np.hstack((test_label1, test_label2))
        test_label = np.hstack((test_label, test_label3))

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map)
    if loader_size != None:
        loader_end = int(x_test.shape[0] * loader_size)
        indices = np.random.choice(x_test.shape[0], loader_end, replace=False)
        x_test = x_test[indices, :, :, :]
        test_label = test_label[indices]

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(np.array(test_label).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return testloader

def test_picked_user(Best_net, testloader, cluster_index, user_index):
    print('\nTesting user:' + str(user_index) + ' in Cluster:' + str(cluster_index))
    Best_net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prob_all = []
    label_all = []
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            combine_input = inputs.squeeze(1)
            # Test the time of processing a feature map
            outputs = Best_net(combine_input)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            prob = outputs.cpu().numpy()
            prob_all.extend(np.argmax(prob, axis=1))
            label = targets.cpu().numpy()
            label_all.extend(label)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Calculate the F1 score
    f1 = f1_score(label_all, prob_all, average='weighted')
    # Save checkpoint.
    acc = 100. * correct / total
    F1 = 100. * f1
    return acc, F1

def train_fine_tune(fine_net, fine_loader_train, fine_loader_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='PyTorch Empatia Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(fine_net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    fine_best_acc = 0
    fine_best_f1 = 0
    for epoch in range(50):
        print('\n fine tune Epoch: %d' % epoch)
        print('\n fine tune Training')
        # CNN definition
        fine_net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(fine_loader_train):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            combine_input = inputs.squeeze(1)

            outputs = fine_net(combine_input)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('\n fine tune Testing')
        fine_net.eval()
        test_loss = 0
        correct = 0
        total = 0
        prob_all = []
        label_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(fine_loader_test):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                combine_input = inputs.squeeze(1)
                # Test the time of processing a feature map
                outputs = fine_net(combine_input)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                prob = outputs.cpu().numpy()
                prob_all.extend(np.argmax(prob, axis=1))
                label = targets.cpu().numpy()
                label_all.extend(label)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Calculate the F1 score
        f1 = f1_score(label_all, prob_all, average='weighted')
        # Save checkpoint.
        acc = 100. * correct / total
        F1 = 100. * f1
        if acc > fine_best_acc:
            fine_best_acc = acc
        if F1 > fine_best_f1:
            fine_best_f1 = F1
        print(' fine tune Best ACC: ------- ' + str(fine_best_acc))
        print(' fine tune Best F1: ------- ' + str(fine_best_f1))
    return fine_best_acc, fine_best_f1

def training_process(net_type, featureMap_stratgy, train_cluster, test_cluster, approach_num):
    parser = argparse.ArgumentParser(description='PyTorch Empatia Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    best_f1 = 0.0

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file,
                                                        train_cluster=train_cluster, test_cluster=test_cluster,
                                                        approach_num=approach_num)
    test_feature_map, test_label = generate_feature_maps(test_array_all, strategy=featureMap_stratgy)
    #
    train_feature_map, train_label = generate_feature_maps(train_array_all, strategy=featureMap_stratgy)
    if featureMap_stratgy == 'All_concat':
        test_feature_map1, test_label1 = generate_feature_maps(test_array_all, strategy='AllFromOne')
        test_feature_map2, test_label2 = generate_feature_maps(test_array_all, strategy='HalfAndHalf')
        test_feature_map3, test_label3 = generate_feature_maps(test_array_all, strategy='HalfAndRandom')
        #
        train_feature_map1, train_label1 = generate_feature_maps(train_array_all, strategy='AllFromOne')
        train_feature_map2, train_label2 = generate_feature_maps(train_array_all, strategy='HalfAndHalf')
        train_feature_map3, train_label3 = generate_feature_maps(train_array_all, strategy='HalfAndRandom')

        test_feature_map = np.dstack((test_feature_map1, test_feature_map2))
        test_feature_map = np.dstack((test_feature_map, test_feature_map3))
        test_label = np.hstack((test_label1, test_label2))
        test_label = np.hstack((test_label, test_label3))
        train_feature_map = np.dstack((train_feature_map1, train_feature_map2))
        train_feature_map = np.dstack((train_feature_map, train_feature_map3))
        train_label = np.hstack((train_label1, train_label2))
        train_label = np.hstack((train_label, train_label3))

    # Normalize the data to 0-255
    # test_feature_map = Normalize0to255(test_feature_map)
    # train_feature_map = Normalize0to255(train_feature_map)
    # test_feature_map = Normalize0to1(test_feature_map)
    # train_feature_map = Normalize0to1(train_feature_map)

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map)
    test_data_int8 = x_test.astype(np.uint8)
    x_train = feature_map_trans(train_feature_map)
    train_data_int8 = x_train.astype(np.uint8)

    # Convert the ndarray to a PyTorch tensor
    x_train_tensor = torch.from_numpy(train_data_int8)
    y_train_tensor = torch.from_numpy(np.array(train_label).astype(int))

    # Create a TensorDataset object
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # Create a DataLoader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(test_data_int8)
    y_test_tensor = torch.from_numpy(np.array(test_label).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Model
    print('==> Building model..')

    net = net_type.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        print('\nTraining')
        # CNN definition
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            combine_input = inputs.squeeze(1)

            outputs = net(combine_input)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        print('\nTesting')
        global best_acc
        global best_f1
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        prob_all = []
        label_all = []
        with torch.no_grad():
            time_list = []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                combine_input = inputs.squeeze(1)
                # Test the time of processing a feature map
                start_time = time.time()
                outputs = net(combine_input)
                end_time = time.time()
                time_consumption = (end_time - start_time) * 1000
                time_list.append(time_consumption)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                prob = outputs.cpu().numpy()
                prob_all.extend(np.argmax(prob, axis=1))
                label = targets.cpu().numpy()
                label_all.extend(label)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print('mean time_consumption (ms): ' + str(np.mean(time_list)))
        # Calculate the F1 score
        f1 = f1_score(label_all, prob_all, average='weighted')
        # Save checkpoint.
        acc = 100. * correct / total
        F1 = 100. * f1
        print('F1:' + str(F1))
        if acc > best_acc:
            best_acc = acc
        if F1 > best_f1:
            best_f1 = F1
        print('Best ACC: ------- ' + str(best_acc))
        print('Best F1: ------- ' + str(best_f1))
        return acc, F1, net

    total_best = 0
    total_f1 = 0
    total_best_net = None
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        now_acc, now_f1, now_net = test(epoch)
        scheduler.step()
        if now_acc > total_best:
            total_best = now_acc
            total_best_net = now_net
        if now_f1 > total_f1:
            total_f1 = now_f1
    return total_best, total_f1, total_best_net


if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'
    approach = 'approach2'

    input_size = 123
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    Combine_model = ParallelCNNLSTMModel(input_size, hidden_size, num_layers, num_classes)

    cluster_funcs = ['test_method1']
    # cluster_funcs = ['test_method2']
    pre_cluster_acc = []
    pre_cluster_f1 = []
    all_cluster_fine_acc = []
    all_cluster_fine_f1 = []
    all_cluster_good_acc = []
    all_cluster_good_f1 = []
    all_cluster_fine_acc_std = []
    all_cluster_fine_f1_std = []
    all_cluster_good_acc_std = []
    all_cluster_good_f1_std = []
    for cluster_func in cluster_funcs:
        # Extract the clusters from .csv
        train_cluster_dict, test_cluster_dict = get_clusters('../partitions_traintest.csv', cluster_func)
        # merge all the validations.
        merged_train = {}
        for key, sub_dict in train_cluster_dict.items():
            for sub_key, sub_list in sub_dict.items():
                if sub_key not in merged_train:
                    merged_train[sub_key] = []
                merged_train[sub_key].extend(sub_list)
        for sub_key in merged_train:
            merged_train[sub_key] = set(merged_train[sub_key])
        merged_test = {}
        for key, sub_dict in test_cluster_dict.items():
            for sub_key, sub_list in sub_dict.items():
                if sub_key not in merged_test:
                    merged_test[sub_key] = []
                merged_test[sub_key].extend(sub_list)
        for sub_key in merged_test:
            merged_test[sub_key] = set(merged_test[sub_key])

        merged_dict = {}
        for key in merged_train:
            merged_dict[key] = merged_train[key].union(merged_test[key])

        for index_cluster, value_cluster in merged_dict.items():
            # Separate train and test
            train_set, test_set = get_train_test_fine(value_cluster)
            set_all = train_set + test_set
            pre_acc_list = []
            pre_f1_list = []

            cluster_fine_acc = []
            cluster_fine_f1 = []
            cluster_good_acc = []
            cluster_good_f1 = []

            # Pick one user from test set
            for test_user in set_all:
                train_set_copy = train_set.copy()
                test_set_copy = test_set.copy()
                # if test_user in train_set_copy:
                #     train_set_copy.remove(test_user)
                # elif test_user in test_set_copy:
                #     test_set_copy.remove(test_user)
                # training
                strategy = 'All_concat'
                print('\nStart_func:' + str(cluster_func))
                print('Start_cluster:' + str(index_cluster))
                print('Picked user Num:' + str(test_user))
                print('Start_strategy: ' + strategy)
                best_acc = 0  # best test accuracy
                best_f1 = 0.0  # best test accuracy
                now_best, now_f1, now_net = training_process(Combine_model, featureMap_stratgy=strategy,
                                                             train_cluster=train_set_copy, test_cluster=test_set_copy,
                                                             approach_num=approach)
                # test the best net in the picked user
                test_loader = Generate_dataloader(data_file=data_file, log_file=log_file, Picked_user=test_user,
                                    featureMap_stratgy=strategy)
                pre_acc, pre_f1 = test_picked_user(now_net, test_loader, cluster_index=index_cluster,
                                                   user_index=test_user)
                pre_acc_list.append(pre_acc)
                pre_f1_list.append(pre_f1)
                # fine tune test user, k fold
                K = 5
                fineAcc_list = []
                fineF1_list = []
                for cross_index in range(K):
                    print('\nStart_func:' + str(cluster_func))
                    print('Start_cluster:' + str(index_cluster))
                    print('Picked user Num:' + str(test_user))
                    print('Start_strategy: ' + strategy)
                    print('Start_Kfold: ' + str(cross_index + 1))
                    fine_loader = Generate_dataloader(data_file=data_file, log_file=log_file, Picked_user=test_user,
                                        featureMap_stratgy=strategy, loader_size=0.3)
                    fine_acc, fine_f1 = train_fine_tune(now_net, fine_loader_test=test_loader,
                                                        fine_loader_train=fine_loader)
                    fineAcc_list.append(fine_acc)
                    fineF1_list.append(fine_f1)
                    print('ACC list:')
                    print(fineAcc_list)
                    print('F1 list:')
                    print(fineF1_list)

                print('\nFunc:' + str(cluster_func))
                print('Cluster:' + str(index_cluster))
                print('Picked user Num:' + str(test_user))
                print('Strategy: ' + strategy)
                print('Performance before fine tune:')
                print('pre acc:' + str(pre_acc))
                print('pre f1:' + str(pre_f1))
                print('bestAcc_list fine tune: ')
                print(fineAcc_list)
                print('Mean best acc:' + str(np.mean(fineAcc_list)))
                cluster_fine_acc.append(np.mean(fineAcc_list))
                print('Mean good acc:' + str(np.max(fineAcc_list)))
                cluster_good_acc.append(np.max(fineAcc_list))
                print('bestF1_list fine tune: ')
                print(fineF1_list)
                print('Mean best F1:' + str(np.mean(fineF1_list)))
                cluster_fine_f1.append(np.mean(fineF1_list))
                print('Mean good F1:' + str(np.max(fineF1_list)))
                cluster_good_f1.append(np.max(fineF1_list))

            print('\nFunc:' + str(cluster_func))
            print('Cluster done:' + str(index_cluster))
            print('Performance before fine tune:')
            print('pre acc average:' + str(np.mean(pre_acc_list)))
            print('pre f1 average:' + str(np.mean(pre_f1_list)))
            print('For cluster: ' + str(index_cluster) + ' calculate performance:')
            print('Mean best acc:' + str(np.mean(cluster_fine_acc)))
            all_cluster_fine_acc.append(np.mean(cluster_fine_acc))
            all_cluster_fine_acc_std.append(np.std(cluster_fine_acc))
            print('Mean good acc:' + str(np.max(cluster_good_acc)))
            all_cluster_good_acc.append(np.max(cluster_good_acc))
            all_cluster_good_acc_std.append(np.std(cluster_good_acc))

            print('Mean best f1:' + str(np.mean(cluster_fine_f1)))
            all_cluster_fine_f1.append(np.mean(cluster_fine_f1))
            all_cluster_fine_f1_std.append(np.std(cluster_fine_f1))
            print('Mean good f1:' + str(np.max(cluster_good_f1)))
            all_cluster_good_f1.append(np.max(cluster_good_f1))
            all_cluster_good_f1_std.append(np.std(cluster_good_f1))


    print('\n')
    print('Mean array:')
    print(all_cluster_fine_acc)
    print(all_cluster_fine_f1)
    for index, value in enumerate(all_cluster_fine_acc):
        print('cluster: ' + str(index))
        print('acc: ' + str(value))
        print('f1: ' + str(all_cluster_fine_f1[index]))
        print('acc std: ' + str(all_cluster_fine_acc_std[index]))
        print('f1 std: ' + str(all_cluster_fine_f1_std[index]))
        print('\n')
    print('Mean value of acc:' + str(np.mean(all_cluster_fine_acc)))
    print('Std value of acc:' + str(np.std(all_cluster_fine_acc)))
    print('Mean value of f1:' + str(np.mean(all_cluster_fine_f1)))
    print('Std value of f1:' + str(np.std(all_cluster_fine_f1)))
    print('\n')

    print('\n')
    print('Good array:')
    print(all_cluster_good_acc)
    print(all_cluster_good_f1)
    for index, value in enumerate(all_cluster_good_acc):
        print('cluster: ' + str(index))
        print('acc: ' + str(value))
        print('f1: ' + str(all_cluster_good_f1[index]))
        print('acc std: ' + str(all_cluster_good_acc_std[index]))
        print('f1 std: ' + str(all_cluster_good_f1_std[index]))
        print('\n')
    print('Mean good of acc:' + str(np.mean(all_cluster_good_acc)))
    print('Std good of acc:' + str(np.std(all_cluster_good_acc)))
    print('Mean good of f1:' + str(np.mean(all_cluster_good_f1)))
    print('Std good of f1:' + str(np.std(all_cluster_good_f1)))
    print('\n')

