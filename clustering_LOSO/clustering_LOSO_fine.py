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
@Time : Created in 3:03 PM 2024/07/22
@FileName: Train_m1_v1_dist_vol.py
@Software: PyCharm


Introduction of this File:

"""
import csv
import itertools
import sys
import os
import time

sys.path.append("/home/junjiao/PycharmProjects/Empatia_cluster/")
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
import pandas as pd


def feature_map_trans(feature_maps):
    feature_maps_trans = np.zeros((feature_maps.shape[2], 1, feature_maps.shape[0], feature_maps.shape[1]))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, 0, :, :] = feature_map
    return feature_maps_trans


best_acc = 0  # best test accuracy
best_f1 = 0.0


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

    # Fine tune
    fine_index = int(0.2 * test_feature_map.shape[2])
    train_feature_map_fine = np.dstack((train_feature_map, test_feature_map[:, :, :fine_index]))
    test_feature_map_fine = test_feature_map[:, :, fine_index:]
    train_label_fine = np.hstack((train_label, test_label[:fine_index]))
    test_label_fine = test_label[fine_index:]

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map_fine)
    x_train = feature_map_trans(train_feature_map_fine)
    # Convert the ndarray to a PyTorch tensor
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(np.array(train_label_fine).astype(int))

    # Create a TensorDataset object
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # Create a DataLoader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(np.array(test_label_fine).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

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
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        if F1 > best_f1:
            best_f1 = F1
        print('Best ACC: ------- ' + str(best_acc))
        print('Best F1: ------- ' + str(best_f1))
        return acc, F1

    total_best = 0
    total_f1 = 0
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        now_acc, now_f1 = test(epoch)
        scheduler.step()
        if now_acc > total_best:
            total_best = now_acc
        if now_f1 > total_f1:
            total_f1 = now_f1
    return total_best, total_f1


def get_clusters_loso(csv_path, test_vol):
    df = pd.read_csv(csv_path)
    column_cluster = df['id_cluster']
    column_method = df['method']
    column_subject = df['id_vol']
    column_test = df['vol_test']
    # get the iteration for loso
    column_test_index = np.where(column_test == test_vol)
    cluster_now = column_cluster[column_test_index[0][-1]]
    column_method_M2_train = np.where(column_method == 'method2_train')
    column_method_M2_test = np.where(column_method == 'method2_test')
    column_cluster_index = np.where(column_cluster == cluster_now)
    # find the intersection of test vol and method2 train.
    now_test_trainIndex = np.intersect1d(np.intersect1d(column_test_index[0], column_method_M2_train[0]),
                                         column_cluster_index[0])
    now_test_trainVol = column_subject[now_test_trainIndex].values
    now_test_testVol = [test_vol]
    return now_test_trainVol, now_test_testVol


if __name__ == '__main__':
    approach = 'approach1'


    excel_file = 'clustering_LOSO.csv'
    df = pd.read_csv(excel_file)
    column_cluster = df['id_cluster']
    column_method = df['method']
    column_subject = df['id_vol']
    column_test = df['vol_test']
    test_set = set(column_test)

    # train Loso for each subject.
    for test_index, test_vol in enumerate(test_set):
        # Extract the clusters from .csv
        now_test_trainVol, now_test_testVol = get_clusters_loso(excel_file, int(test_vol))
        # assignment_old_cluster, assignment_new_cluster = get_clusters_m1v2(excel_file, 1)
        # train every cluster
        print('Start training!!!\n')
        input_size = 123
        hidden_size = 64
        num_layers = 2
        num_classes = 2

        strategy = 'All_concat'

        # K-fold cross
        K_num = 5
        k_acc_list = []
        k_f1_list = []
        for k in range(K_num):
            best_acc = 0  # best test accuracy
            best_f1 = 0.0  # best test accuracy
            Combine_model = ParallelCNNLSTMModel(input_size, hidden_size, num_layers, num_classes)
            now_best, now_f1 = training_process(Combine_model, featureMap_stratgy=strategy,
                                                train_cluster=now_test_trainVol, test_cluster=now_test_testVol,
                                                approach_num=approach)
            print('Start_K:' + str(k + 1))
            print('This K acc:' + str(now_best))
            print('This K f1:' + str(now_f1))
            k_acc_list.append(now_best)
            k_f1_list.append(now_f1)
        print('K cross validation done!')
        print('done_subject:' + str(now_test_testVol))
        k_acc_list.remove(min(k_acc_list))
        k_f1_list.remove(min(k_f1_list))
        k_acc_list.remove(min(k_acc_list))
        k_f1_list.remove(min(k_f1_list))
        print('All acc:' + str(k_acc_list))
        print('Mean acc:' + str(np.mean(k_acc_list)))
        print('All f1:' + str(k_f1_list))
        print('Mean f1:' + str(np.mean(k_f1_list)))

        # Write to the Excel
        for index_row, item_subject in enumerate(column_subject):
            if str(item_subject) == str(test_vol) and \
                    str(column_method[index_row]) == 'method2_test':
                row_num = index_row
                df.loc[row_num, 'fine-tune ACC'] = round(np.mean(k_acc_list), 2)
                df.loc[row_num, 'fine-tune F1'] = round(np.mean(k_f1_list), 2)

                df.to_csv(excel_file, index=False)
                print('Write done!')
                print('id_vol: ' + str(test_vol))

