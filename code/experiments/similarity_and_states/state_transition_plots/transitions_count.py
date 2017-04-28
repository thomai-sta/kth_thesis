#!/usr/local/bin/python
#  -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')

n_components = []
MCI_nonAD_diag = []
MCI_nonAD_inter = []
MCI_total_trans_nonAD = []
MCI_AD_diag = []
MCI_AD_inter = []
MCI_total_trans_AD = []
MCI_total_trans = []
non_MCI_nonAD_diag = []
non_MCI_nonAD_inter = []
non_MCI_total_trans_nonAD = []
non_MCI_AD_diag = []
non_MCI_AD_inter = []
non_MCI_total_trans_AD = []
non_MCI_total_trans = []


# Read all the lines, then strip the 'new line' character and split into the
# different variables. Then convert to int
file = open("transition_count", 'r')
lines = file.readlines()
lines = lines[1:]
for line in lines:
    n_components_tmp, MCI_nonAD_diag_tmp, MCI_nonAD_inter_tmp,\
        MCI_total_trans_nonAD_tmp, MCI_AD_diag_tmp, MCI_AD_inter_tmp,\
        MCI_total_trans_AD_tmp, MCI_total_trans_tmp, non_MCI_nonAD_diag_tmp,\
        non_MCI_nonAD_inter_tmp, non_MCI_total_trans_nonAD_tmp,\
        non_MCI_AD_diag_tmp, non_MCI_AD_inter_tmp, non_MCI_total_trans_AD_tmp,\
        non_MCI_total_trans_tmp = line.split(",")
    n_components.append(int(n_components_tmp))
    MCI_nonAD_diag.append(int(MCI_nonAD_diag_tmp))
    MCI_nonAD_inter.append(int(MCI_nonAD_inter_tmp))
    MCI_total_trans_nonAD.append(int(MCI_total_trans_nonAD_tmp))
    MCI_AD_diag.append(int(MCI_AD_diag_tmp))
    MCI_AD_inter.append(int(MCI_AD_inter_tmp))
    MCI_total_trans_AD.append(int(MCI_total_trans_AD_tmp))
    MCI_total_trans.append(int(MCI_total_trans_tmp))
    non_MCI_nonAD_diag.append(int(non_MCI_nonAD_diag_tmp))
    non_MCI_nonAD_inter.append(int(non_MCI_nonAD_inter_tmp))
    non_MCI_total_trans_nonAD.append(int(non_MCI_total_trans_nonAD_tmp))
    non_MCI_AD_diag.append(int(non_MCI_AD_diag_tmp))
    non_MCI_AD_inter.append(int(non_MCI_AD_inter_tmp))
    non_MCI_total_trans_AD.append(int(non_MCI_total_trans_AD_tmp))
    non_MCI_total_trans.append(int(non_MCI_total_trans_tmp))


file.close()


plt.figure()
plt.title("MCI Subject-Initial-Group, CN/MCI Subject-End-Group", fontsize=18)
plt.xlabel('Number of States', fontsize=18)
plt.ylabel('Transition Count', fontsize=18)

plt.hold(True)
plt.grid()

plt.plot(n_components, MCI_nonAD_diag, label="Same State Transitions",
         linewidth=2)
plt.plot(n_components, MCI_nonAD_inter, label="Inter-State Transitions",
         linewidth=2)
plt.plot(n_components, MCI_total_trans_nonAD,
         label="CN/MCI Subject-End-Group Total Transitions",
         linewidth=2)
plt.plot(n_components, MCI_total_trans,
         label="MCI Subject-Initial-Group Total Transitions",
         linewidth=2)

plt.legend()

MCI_nonAD_diag_percent = np.array(MCI_nonAD_diag, dtype=np.float) /\
    np.array(MCI_total_trans_nonAD, dtype=np.float)
MCI_nonAD_diag_std = np.std(MCI_nonAD_diag_percent)
MCI_nonAD_diag_mean = np.mean(MCI_nonAD_diag_percent)
print("MCI non-AD Diag: %f, %f" % (MCI_nonAD_diag_mean, MCI_nonAD_diag_std))
MCI_nonAD_inter_percent = np.array(MCI_nonAD_inter, dtype=np.float) /\
    np.array(MCI_total_trans_nonAD, dtype=np.float)
MCI_nonAD_inter_std = np.std(MCI_nonAD_inter_percent)
MCI_nonAD_inter_mean = np.mean(MCI_nonAD_inter_percent)
print("MCI non-AD inter: %f, %f" % (MCI_nonAD_inter_mean, MCI_nonAD_inter_std))
# print(MCI_nonAD_diag_percent)
# print(MCI_nonAD_inter_percent)

plt.figure()
plt.title("MCI Subject-Initial-Group, AD Subject-End-Group", fontsize=18)
plt.xlabel('Number of States', fontsize=18)
plt.ylabel('Transition Count', fontsize=18)
plt.hold(True)
plt.grid()

plt.plot(n_components, MCI_AD_diag, label="Same State Transitions",
         linewidth=2)
plt.plot(n_components, MCI_AD_inter, label="Inter-State Transitions",
         linewidth=2)
plt.plot(n_components, MCI_total_trans_AD,
         label="AD Subject-End-Group Total Transitions",
         linewidth=2)
plt.plot(n_components, MCI_total_trans,
         label="MCI Subject-Initial-Group Total Transitions",
         linewidth=2)

plt.legend()

MCI_AD_diag_percent = np.array(MCI_AD_diag, dtype=np.float) /\
    np.array(MCI_total_trans_AD, dtype=np.float)
MCI_AD_diag_std = np.std(MCI_AD_diag_percent)
MCI_AD_diag_mean = np.mean(MCI_AD_diag_percent)
print("MCI AD diag: %f, %f" % (MCI_AD_diag_mean, MCI_AD_diag_std))
MCI_AD_inter_percent = np.array(MCI_AD_inter, dtype=np.float) /\
    np.array(MCI_total_trans_AD, dtype=np.float)
MCI_AD_inter_std = np.std(MCI_AD_inter_percent)
MCI_AD_inter_mean = np.mean(MCI_AD_inter_percent)
print("MCI AD inter: %f, %f" % (MCI_AD_inter_mean, MCI_AD_inter_std))
# print(MCI_AD_diag_percent)
# print(MCI_AD_inter_percent)

plt.figure()
plt.title("CN/AD Subject-Initial-Group, CN/MCI Subject-End-Group", fontsize=18)
plt.xlabel('Number of States', fontsize=18)
plt.ylabel('Transition Count', fontsize=18)
plt.hold(True)
plt.grid()

plt.plot(n_components, non_MCI_nonAD_diag, label="Same State Transitions",
         linewidth=2)
plt.plot(n_components, non_MCI_nonAD_inter, label="Inter-State Transitions",
         linewidth=2)
plt.plot(n_components, non_MCI_total_trans_nonAD,
         label="CN/MCI Subject-End-Group Total Transitions", linewidth=2)
plt.plot(n_components, non_MCI_total_trans,
         label="CN/AD Subject-Initial-Group Total Transitions", linewidth=2)

plt.legend()

non_MCI_nonAD_diag_percent = np.array(non_MCI_nonAD_diag, dtype=np.float) /\
    np.array(non_MCI_total_trans_nonAD, dtype=np.float)
non_MCI_nonAD_diag_std = np.std(non_MCI_nonAD_diag_percent)
non_MCI_nonAD_diag_mean = np.mean(non_MCI_nonAD_diag_percent)
print("non-MCI non-AD diag: %f, %f" % (non_MCI_nonAD_diag_mean,
                                       non_MCI_nonAD_diag_std))

non_MCI_nonAD_inter_percent = np.array(non_MCI_nonAD_inter, dtype=np.float) /\
    np.array(non_MCI_total_trans_nonAD, dtype=np.float)
non_MCI_nonAD_inter_std = np.std(non_MCI_nonAD_inter_percent)
non_MCI_nonAD_inter_mean = np.mean(non_MCI_nonAD_inter_percent)
print("non-MCI non-AD inter: %f, %f" % (non_MCI_nonAD_inter_mean,
                                        non_MCI_nonAD_inter_std))
# print(non_MCI_nonAD_diag_percent)
# print(non_MCI_nonAD_inter_percent)

plt.figure()
plt.title("CN/AD Subject-Initial-Group, AD Subject-End-Group", fontsize=18)
plt.xlabel('Number of States', fontsize=18)
plt.ylabel('Transition Count', fontsize=18)
plt.hold(True)
plt.grid()

plt.plot(n_components, non_MCI_AD_diag, label="Same State Transitions",
         linewidth=2)
plt.plot(n_components, non_MCI_AD_inter, label="Inter-State Transitions",
         linewidth=2)
plt.plot(n_components, non_MCI_total_trans_AD,
         label="AD Subject-End-Group Total Transitions",
         linewidth=2)
plt.plot(n_components, non_MCI_total_trans,
         label="CN/AD Subject-Initial-Group Total Transitions", linewidth=2)

plt.legend()

non_MCI_AD_diag_percent = np.array(non_MCI_AD_diag, dtype=np.float) /\
    np.array(non_MCI_total_trans_AD, dtype=np.float)
non_MCI_AD_diag_std = np.std(non_MCI_AD_diag_percent)
non_MCI_AD_diag_mean = np.mean(non_MCI_AD_diag_percent)
print("non-MCI AD diag: %f, %f" % (non_MCI_AD_diag_mean,
                                   non_MCI_AD_diag_std))
non_MCI_AD_inter_percent = np.array(non_MCI_AD_inter, dtype=np.float) /\
    np.array(non_MCI_total_trans_AD, dtype=np.float)
non_MCI_AD_inter_std = np.std(non_MCI_AD_inter_percent)
non_MCI_AD_inter_mean = np.mean(non_MCI_AD_inter_percent)
print("non-MCI AD inter: %f, %f" % (non_MCI_AD_inter_mean,
                                    non_MCI_AD_inter_std))
# print(non_MCI_AD_diag_percent)
# print(non_MCI_AD_inter_percent)

plt.show()
