#!/usr/local/bin/python
#!/usr/bin/python

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

""" FOLLOW_UPS: Number of follow up instances to include
    UP_TO:      Variable indicating if we'll use instances of patients with the
                exact number of follow ups as FOLLOW_UPS or also use patients
                with fewer follow ups """

dataset = np.recfromcsv('/home/thomai/Dropbox/thesis_KTH_KI/python/dataset.csv')

""" TAKE OUT FLAGGED SUBJECT! """
flagged_idx = 1216
flagged_subjects = dataset['sid'][flagged_idx]
dataset = dataset[dataset['sid'] != flagged_subjects]

# CREATE SETS OF KEYS
# ID info col 1:6
keys_ID = ['iid', 'tempid', 'temptp', 'tp', 'rid', 'sid']
# area info col 7:40
keys_area = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal',
             'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal',
             'inferiortemporal', 'isthmuscingulate', 'lateraloccipital',
             'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
             'middletemporal', 'parahippocampal', 'paracentral',
             'parsopercularis', 'parsorbitalis', 'parstriangularis',
             'pericalcarine', 'postcentral', 'posteriorcingulate',
             'precentral', 'precuneus', 'rostralanteriorcingulate',
             'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal',
             'superiortemporal', 'supramarginal', 'frontalpole', 'temporalpole',
             'transversetemporal', 'insula']
problem = False

for key in keys_area:
    check = dataset[key]
    test = np.where(check < 0)
    if np.any(test):
        problem = True
        print("Problem with %s, in places: %s" %(key, test))

if not problem:
    print("AREA FEATURES GOOD")
else:
    problem = False

# volume info col 41:61
keys_volume = ['lateralventricle', 'inflatvent', 'cerebellumwhitematter',
               'cerebellumcortex', 'thalamusproper', 'caudate', 'putamen',
               'pallidum', 'ventricle_3rd', 'ventricle_4th', 'brainstem',
               'hippocampus', 'amygdala', 'csf', 'accumbensarea', 'ventraldc',
               'cc_posterior', 'cc_mid_posterior', 'cc_central',
               'cc_mid_anterior', 'cc_anterior']

for key in keys_volume:
    check = dataset[key]
    test = np.where(check < 0)
    if np.any(test):
        problem = True
        print("Problem with %s, in places: %s" %(key, test))

if not problem:
    print("VOLUME FEATURES GOOD")
else:
    problem = False

# volume normalizer col 62
keys_normalizer = ['icv']
for key in keys_normalizer:
    check = dataset[key]
    test = np.where(check < 0)
    if np.any(test):
        problem = True
        print("Problem with %s, in places: %s" %(key, test))

if not problem:
    print("NORMALIZER GOOD")
else:
    problem = False

# personal info col 63:73
keys_personal = ['age', 'weight', 'apoe1', 'apoe2', 'cdr', 'npi', 'mmse',
                 'faq', 'gds', 'sex', 'gender']

# diagnosis info col 74:end
keys_diagnosis = ['dxgroup', 'dxgr', 'dxcurr', 'dxcur', 'dxconv', 'dxcon']
keys_diagnosis_strings = ['dxgroup', 'dxcurr', 'dxconv']
keys_diagnosis_ints = ['dxgr', 'dxcur', 'dxcon']
for key in keys_diagnosis_strings:
    check = dataset[key]
    if key == 'dxgroup' or key == 'dxcurr':
        test_a = (check == 'hc')
        test_b = (check == 'mci')
        test_c = (check == 'ad')
        test_d = test_a + test_b + test_c
        test = np.where(test_d == False)
    elif key == 'dxconv':
        test_a = (check == 'no')
        test_b = (check == 'rev')
        test_c = (check == 'conv')
        test_d = test_a + test_b + test_c
        test = np.where(test_d == False)
    if np.any(test):
        problem = True
        print("Problem with %s, in places: %s" %(key, test))
if not problem:
    print("DIAGNOSIS STRINGS GOOD")
else:
    problem = False

for key in keys_diagnosis_ints:
    check = dataset[key]
    if key == 'dxgr' or key == 'dxcon':
        test_a = (check == 0)
        test_b = (check == 1)
        test_c = (check == 2)
        test_d = test_a + test_b + test_c
        test = np.where(test_d == False)
    elif key == 'dxcur':
        test_a = (check == 1)
        test_b = (check == 2)
        test_c = (check == 3)
        test_d = test_a + test_b + test_c
        test = np.where(test_d == False)
    if np.any(test):
        problem = True
        print("Problem with %s, in places: %s" %(key, test))
if not problem:
    print("DIAGNOSIS INTS GOOD")
else:
    problem = False

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "keys.pickle"), "wb") as f:
    pickle.dump([keys_ID, keys_area, keys_volume, keys_normalizer,
                 keys_personal], f)

""" STRING TO INT FEATURES
     temptp contains the total length of the longitudinal MRI set:
       - m12 = one follow-up
       - m24 = two follow-ups
       - m36 = three follow-ups
     These get converted to integers:
       - m12 --> 1
       - m24 --> 2
       - m36 --> 3 """
temp = dataset['temptp']
temptp = np.empty(temp.shape, dtype=temp[0].dtype)
for i in range(0, len(temp)):
    temptp[i] = temp[i].lower()

a = np.unique(temptp)
a = np.sort(a)
temptp_int = np.empty(temptp.shape, dtype=int)
for i, c in enumerate(a):
    np.place(temptp_int, temptp == c, i + 1)

del (temp)
del (a)

""" tp contains the current follow-up of the longitudinal MRI set:
       - m00 = cross-sectional
       - m12 = first follow-up
       - m24 = second follow-up
       - m36 = third follow-up
    These get converted to integers:
       - m00 --> 0
       - m12 --> 1
       - m24 --> 2
       - m36 --> 3 """
temp = dataset['tp']
tp = np.empty(temp.shape, dtype=temp[0].dtype)
for i in range(0, len(temp)):
    tp[i] = temp[i].lower()

a = np.unique(tp)
a = np.sort(a)
tp_int = np.empty(temptp.shape, dtype=int)
for i, c in enumerate(a):
    np.place(tp_int, tp == c, i)


area_feats = dataset[keys_area[0]]

for i in range(1, len(keys_area)):
    area_feats = np.vstack((area_feats, dataset[keys_area[i]]))
area_feats = np.transpose(area_feats)

""" NORMALIZE VOLUME FEATURES """
ICV = dataset['icv']
vol_feats = dataset[keys_volume[0]] / ICV
for i in range(1, len(keys_volume)):
    vol_feats = np.vstack((vol_feats, dataset[keys_volume[i]] / ICV))
vol_feats = np.transpose(vol_feats)

""" - First go over subjects
    - For every subject go over instances
    - For every subject keep dxgroup and endgroup """
sid = dataset['sid']
subjects = np.unique(sid)
dxgroup = dataset['dxgr']
# Change numbering
idx_hc = (dxgroup == 0)
idx_mci = (dxgroup == 2)
idx_ad = (dxgroup == 1)

dxgroup[idx_mci] = 1
dxgroup[idx_ad] = 2

dxcurrent = dataset['dxcur']
dxcurrent = dxcurrent - 1

observations = np.empty(shape=(0, area_feats.shape[1] + vol_feats.shape[1]))
subject_group = []
subject_end_group = []
scan_group = []
lengths = []
for subject_idx, subject_sid in enumerate(subjects):
    sid_idx = np.where(sid == subject_sid)
    sid_idx = np.array(sid_idx)
    sid_idx = sid_idx.T

    lengths.append(len(sid_idx))
    subject_group.append(dxgroup[sid_idx[0, 0]])
    # Each subject is an emission sequence.
    # Need to keep up with the length of each sequence
    last_dx = -1
    for instance in np.arange(len(sid_idx)):
        for s in sid_idx:
            if tp_int[s] == instance:
                if instance == 1 and dxcurrent[s[0]] == 1 and dxgroup[s[0]] == 2:
                    print instance, dxcurrent[s[0]], dxgroup[s[0]], subject_sid
                temp = np.hstack((area_feats[s[0]], vol_feats[s[0]]))
                observations = np.vstack((observations, temp))
                scan_group.append(dxcurrent[s[0]])
                last_dx = dxcurrent[s[0]]
    subject_end_group.append(last_dx)



lengths = np.array(lengths)
subject_group = np.array(subject_group)
subject_end_group = np.array(subject_end_group)
scan_group = np.array(scan_group)

""" Reshape scan_group """
subject_idxs = []
state_diagnosis = np.zeros((0, np.max(lengths)))
start = 0
end = lengths[0]
for i in range(len(lengths)):
    subject_idxs.append(start)
    state_diagnosis = np.vstack((state_diagnosis, -np.ones((1, np.max(lengths)))))
    state_diagnosis[-1, :len(scan_group[start:end])] = scan_group[start:end]
    start += lengths[i]
    if i < len(lengths) - 1:
        end += lengths[i + 1]

subject_idxs = np.array(subject_idxs)

""" Save data """
here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, "data/for_hmm.pickle"), "wb") as f:
    pickle.dump([observations, lengths, subject_group, subject_end_group,
                 scan_group, state_diagnosis, subject_idxs], f)

# """ Plot states' diagnoses """
# colours = ['g', 'm', 'r']
# plt.title("State diagnoses of different groups")
# for i, diag in enumerate(state_diagnosis):
#     x = np.arange(lengths[i])
#     if subject_group[i] == 0:
#         # Healthy group
#         plt.subplot(311)
#         plt.title('Healthy group: %d' % (np.sum(subject_group == 0)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_group[i]])
#     elif subject_group[i] == 1:
#         # MCI group
#         plt.subplot(312)
#         plt.title('MCI group: %d' % (np.sum(subject_group == 1)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_group[i]])
#     elif subject_group[i] == 2:
#         # AD group
#         plt.subplot(313)
#         plt.title('AD group: %d' % (np.sum(subject_group == 2)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_group[i]])
#
# plt.subplot(311)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
# plt.subplot(312)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
# plt.subplot(313)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
#
# plt.figure()
# plt.title("End Groups")
# for i, diag in enumerate(state_diagnosis):
#     x = np.arange(lengths[i])
#     if subject_end_group[i] == 0:
#         # Healthy group
#         plt.subplot(311)
#         plt.title('Healthy "end" group: %d' % (np.sum(subject_end_group == 0)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_end_group[i]])
#     elif subject_end_group[i] == 1:
#         # MCI group
#         plt.subplot(312)
#         plt.title('MCI "end" group: %d' % (np.sum(subject_end_group == 1)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_end_group[i]])
#     elif subject_end_group[i] == 2:
#         # AD group
#         plt.subplot(313)
#         plt.title('AD "end" group: %d' % (np.sum(subject_end_group == 2)))
#         plt.plot(x, diag[:lengths[i]], colours[subject_end_group[i]])
#
#
# plt.subplot(311)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
# plt.subplot(312)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
# plt.subplot(313)
# plt.axis([0, 3, 0, np.max(np.unique(state_diagnosis))])
# plt.show()