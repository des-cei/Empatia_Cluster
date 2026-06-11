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
@Time : Created in 1:01 PM 2024/05/24   
@FileName: read_log.py                           
@Software: PyCharm


Introduction of this File:

"""
import numpy as np

if __name__ == '__main__':
    file = "fine_tune_approach2_T1.log"
    test = ''
    if 'T1' in file:
        test = 'T1'
    elif 'T2' in file:
        test = 'T2'
    cluster_acc = []
    cluster_f1 = []
    valid_acc = []
    valid_f1 = []
    with open(file, "r") as f:  # 打开文件
        lines = f.readlines()
        for index, line in enumerate(lines):
            if 'average good acc in cluster' in line:
                cluster = str(int(lines[index].split(' -- ')[0][-1]) + 1)
                a = (lines[index].split(' -- '))[1]
                acc = round(float((lines[index].split(' -- '))[1]), 2)

                acc_std = round(float((lines[index + 1].split(' -- '))[1]), 2)
                f1 = round(float((lines[index + 2].split(' -- '))[1]), 2)

                f1_std = round(float((lines[index + 3].split(' -- '))[1]), 2)
                print(test + ' & - & ' + cluster + ' & ' + str(acc) + ' & ' + str(acc_std) + ' & ' +
                      str(f1) + ' & ' + str(f1_std) + ' \\\\')

                cluster_acc.append(float(acc))
                cluster_f1.append(float(f1))
            elif 'average good acc in valid' in line:
                valid = lines[index].split(' -- ')[0].split(':')[-1]
                a = (lines[index].split(' -- '))[1]
                acc = round(float((lines[index].split(' -- '))[1]), 2)
                acc_std = round(float((lines[index + 1].split(' -- '))[1]), 2)
                f1 = round(float((lines[index + 2].split(' -- '))[1]), 2)
                f1_std = round(float((lines[index + 3].split(' -- '))[1]), 2)
                print(test + ' & ' + valid + ' & - & ' + str(acc) + ' & ' + str(acc_std) + ' & ' +
                      str(f1) + ' & ' + str(f1_std) + ' \\\\')

                valid_acc.append(float(acc))
                valid_f1.append(float(f1))
    print('cluster acc mean:' + str(round(np.mean(cluster_acc), 2)))
    print('cluster f1 mean:' + str(round(np.mean(cluster_f1), 2)))
    print('cluster acc std:' + str(round(np.std(cluster_acc), 2)))
    print('cluster f1 std:' + str(round(np.std(cluster_f1), 2)))
    print('valid acc mean:' + str(round(np.mean(valid_acc), 2)))
    print('valid f1 mean:' + str(round(np.mean(valid_f1), 2)))
    print('valid acc std:' + str(round(np.std(valid_acc), 2)))
    print('valid f1 std:' + str(round(np.std(valid_f1), 2)))


