import json
import sys
sys.path.append("/home/junjiao/PycharmProjects/Empatia_cluster/")
import tensorflow as tf
from sklearn.metrics import f1_score
from Feature_wise_Normalization.Create_feature_maps import get_normalization, generate_feature_maps

# assert float(tf.__version__[:3]) >= 2.3
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

from random import shuffle
import csv
from collections import defaultdict
from Feature_wise_Normalization.Data_extraction import Abnormal_detection
from Feature_wise_Normalization.Data_normalization import Normalize_data

def get_clusters(csv_path, cluster_func):
    with open(csv_path, newline='') as csvfile:
        spamreader_test = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # test dict creation
        test_all_user = []
        test_all_cluster = []
        test_all_validation = []
        all_row = []
        for row_test in spamreader_test:
            row_split = row_test[0].split(';')
            all_row.append(row_test)
            if cluster_func in row_test[0]:
                test_all_user.append(int(row_split[0].replace('"', '')))
                test_all_cluster.append(int(row_split[1]))
                test_all_validation.append(int(row_split[2]))
        test_cluster_array = np.vstack((np.array(test_all_user), np.array(test_all_cluster),
                                        np.array(test_all_validation)))
        # transfer the array into a dict
        test_cluster_dict = defaultdict(set)
        for index in range(test_cluster_array.shape[1]):
            # cluster_dict[cluster_array[1][index]] = cluster_array[1][index]
            test_cluster_dict[test_cluster_array[2][index]].add(test_cluster_array[1][index])

        three_layer_dict = {}

        for outer_key, inner_set in test_cluster_dict.items():
            inner_dict = {}
            for inner_key in inner_set:
                inner_dict[inner_key] = []
                for index in range(test_cluster_array.shape[1]):
                    if test_cluster_array[2][index] == outer_key and test_cluster_array[1][index] == inner_key:
                        inner_dict[inner_key].append(test_cluster_array[0][index])
            three_layer_dict[outer_key] = inner_dict
        test_cluster_dict = three_layer_dict

        # train dict creation
        train_all_user = []
        train_all_cluster = []
        train_all_validation = []
        for row_train in all_row:
            row_split = row_train[0].split(';')
            if 'train' in row_train[0]:
                train_all_user.append(int(row_split[0].replace('"', '')))
                train_all_cluster.append(int(row_split[1]))
                train_all_validation.append(int(row_split[2]))
        train_cluster_array = np.vstack((np.array(train_all_user), np.array(train_all_cluster),
                                        np.array(train_all_validation)))
        # transfer the array into a dict
        train_cluster_dict = defaultdict(set)
        for index in range(train_cluster_array.shape[1]):
            # cluster_dict[cluster_array[1][index]] = cluster_array[1][index]
            train_cluster_dict[train_cluster_array[2][index]].add(train_cluster_array[1][index])

        three_layer_dict = {}

        for outer_key, inner_set in train_cluster_dict.items():
            inner_dict = {}
            for inner_key in inner_set:
                inner_dict[inner_key] = []
                for index in range(train_cluster_array.shape[1]):
                    if train_cluster_array[2][index] == outer_key and train_cluster_array[1][index] == inner_key:
                        inner_dict[inner_key].append(train_cluster_array[0][index])
            three_layer_dict[outer_key] = inner_dict
        train_cluster_dict = three_layer_dict

        return train_cluster_dict, test_cluster_dict

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


def feature_map_trans_tf(feature_maps):
    # keras using shape of (b, w, h, c)
    feature_maps_trans = np.zeros((feature_maps.shape[2], feature_maps.shape[0], feature_maps.shape[1], 1))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, :, :, 0] = feature_map
    return feature_maps_trans


def data_prepare_tf(train_cluster, test_cluster, approach_num, test_user, featureMap_stratgy='All_concat',
                    ifShow=True):
    # Data
    # print('==> Preparing data..')

    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file,
                                                        train_cluster=train_cluster, test_cluster=test_cluster,
                                                        approach_num=approach_num)
    # Save the number of participants in train and test respectively
    train_set = set(train_array_all[124, :])
    test_set = set(test_array_all[124, :])
    participant_dic = {'train_set': train_set, 'test_set': test_set}
    if ifShow == True:
        print('Now split of train and test:')
        print(participant_dic)

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
    test_feature_map = Normalize0to255(test_feature_map)
    train_feature_map = Normalize0to255(train_feature_map)

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans_tf(test_feature_map)
    x_train = feature_map_trans_tf(train_feature_map)
    return x_train, train_label, x_test, test_label


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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


def Merge_dict(cluster_path, cluster_func):
    # Extract the clusters from .csv
    train_cluster_dict, test_cluster_dict = get_clusters(cluster_path, cluster_func)
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

    return merged_dict


def convert_model(model_convert, train_set_convert, test_set_convert, approach_convert, test_user_convert):
    def representative_data_gen():
        x_train, train_label, x_test, test_label = data_prepare_tf(train_cluster=train_set_convert,
                                                                   test_cluster=test_set_convert,
                                                                   approach_num=approach_convert,
                                                                   test_user=test_user_convert, ifShow=False)
        a = []
        for i in range(16):
            img = x_test[i, :, :, :]
            img = img / 255.0
            img = img.astype(np.float32)
            a.append(img)
        a = np.array(a)
        img = tf.data.Dataset.from_tensor_slices(a).batch(1)
        for i in img.take(BATCH_SIZE):
            yield [i]

    converter = tf.lite.TFLiteConverter.from_keras_model(model_convert)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare
    # it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    return tflite_model


def General_training(train_set_gen, test_set_gen, approach_gen, test_user_gen, epoch_num):
    # General training
    x_train, train_label, x_test, test_label = data_prepare_tf(train_cluster=train_set_gen,
                                                               test_cluster=test_set_gen,
                                                               approach_num=approach_gen, test_user=test_user_gen)
    y_train = np.asarray(train_label).astype('uint8').reshape((-1, 1))
    y_test = np.asarray(test_label).astype('uint8').reshape((-1, 1))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=0)
    train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    val_generator = datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)

    # Create the base base_model = tf.keras.applications.resnet.ResNet50(input_shape=IMG_SHAPE,
    # include_top=False, weights=None) base_model = tf.keras.applications.resnet.ResNet152(
    # input_shape=IMG_SHAPE, include_top=False, weights=None)
    base_model = tf.keras.applications.resnet.ResNet101(input_shape=IMG_SHAPE, include_top=False,
                                                        weights=None)
    # base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,
    # weights=None) base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
    # include_top=False, weights=None) base_model = tf.keras.applications.DenseNet201(
    # input_shape=IMG_SHAPE, include_top=False, weights=None)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc', f1_m, precision_m, recall_m])

    # model.summary()

    # print('Number of trainable weights = {}'.format(len(model.trainable_weights)))

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epoch_num,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        verbose=2)

    loss, accuracy, f1_score, precision, recall = model.evaluate(val_generator, verbose=0)
    print('acc: ' + str(accuracy))
    print('f1_score: ' + str(f1_score))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))

    return model


def FineTuning_training(model_fine, Fine_train_gen, Fine_test_gen, epoch_num):
    history = model.fit(Fine_train_gen,
                        steps_per_epoch=len(Fine_train_gen),
                        epochs=epoch_num,
                        validation_data=Fine_test_gen,
                        validation_steps=len(Fine_test_gen),
                        verbose=2)

    loss, accuracy, f1_score, precision, recall = model.evaluate(Fine_test_gen, verbose=0)
    print('acc: ' + str(accuracy))
    print('f1_score: ' + str(f1_score))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))

    return model_fine


def Generate_generator(data_file, log_file, Picked_user, featureMap_stratgy, loader_size=None):
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
    x_test = feature_map_trans_tf(test_feature_map)
    if loader_size != None:
        loader_end = int(x_test.shape[0] * loader_size)
        indices = np.random.choice(x_test.shape[0], loader_end, replace=False)
        x_test = x_test[indices, :, :, :]
        test_label = test_label[indices]

    y_test = np.asarray(test_label).astype('uint8').reshape((-1, 1))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_generator = datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)

    return val_generator


if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'
    approach = 'approach2'
    cluster_funcs = ['test_method1']

    IMAGE_SIZE = 123
    BATCH_SIZE = 32
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
    classes = ('not_fear', 'fear')

    cluster_csv = '../partitions_traintest.csv'

    for cluster_func in cluster_funcs:
        merged_dict = Merge_dict(cluster_csv, cluster_func)

        for index_cluster, value_cluster in merged_dict.items():
            # Separate train and test
            # train_set, test_set = get_train_test_fine(value_cluster)

            # Pick one user from set, then use remain to train
            for test_user in value_cluster:
                train_set = list(value_cluster)
                train_set.remove(test_user)
                train_set, test_set = get_train_test_fine(train_set)
                # training
                strategy = 'All_concat'
                print('\nStart_func:' + str(cluster_func))
                print('Start_cluster:' + str(index_cluster))
                print('Picked user Num:' + str(test_user))
                print('Start_strategy: ' + strategy)
                # General training
                print('General training start!')
                model = General_training(train_set_gen=train_set, test_set_gen=test_set,
                                         approach_gen=approach, test_user_gen=test_user, epoch_num=100)

                # convert model
                tflite_model_general = convert_model(model_convert=model, train_set_convert=train_set,
                                                     test_set_convert=test_set, approach_convert=approach,
                                                     test_user_convert=test_user)
                print('General training model convert done\n')

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                with open('./checkpoint/General_ClusterID_' + str(index_cluster) + '_UserID_' +
                          str(test_user) + '_quant.tflite', 'wb') as f:
                    f.write(tflite_model_general)

                # Fine-tuning Process
                print('Fine tune training start!')
                Fine_train = Generate_generator(data_file=data_file, log_file=log_file, Picked_user=test_user,
                                                featureMap_stratgy=strategy, loader_size=0.3)
                Fine_test = Generate_generator(data_file=data_file, log_file=log_file, Picked_user=test_user,
                                               featureMap_stratgy=strategy)

                model_fine = FineTuning_training(model_fine=model, Fine_train_gen=Fine_train,
                                                 Fine_test_gen=Fine_test, epoch_num=100)
                # convert model
                tflite_model_fine = convert_model(model_convert=model_fine, train_set_convert=train_set,
                                                  test_set_convert=test_set, approach_convert=approach,
                                                  test_user_convert=test_user)
                print('Fine tune training model convert done\n')

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                with open('./checkpoint/FineTuning_ClusterID_' + str(index_cluster) + '_UserID_' +
                          str(test_user) + '_quant.tflite', 'wb') as f:
                    f.write(tflite_model_general)
