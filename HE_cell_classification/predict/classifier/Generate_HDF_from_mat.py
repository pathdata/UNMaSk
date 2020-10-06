import scipy.io as sio
import numpy as np
import h5py
import os
import pathlib
import glob

save_path = os.path.normpath(str(pathlib.Path(r"D:\August2018\cell_classification_HE\2020_ENSClassifier_TA_DUKE\DataS")))
#main_file_path = os.path.normpath(r"D:\Priya_FDrive\Training_HE\partition\Data_HE8")
main_file_path = os.path.normpath(r"D:\Priya_FDrive\Training_HE\partition\Data_HE_Single")

Train_Data_name = 'TrainData181026_DUKE.h5'
Valid_Data_name = 'ValidData181026_DUKE.h5'

#Train_indices = indices_workspace['Train_indices']
Train_files = sorted(glob.glob(os.path.join(main_file_path, 'Training', '*.mat')))
Valid_files = sorted(glob.glob(os.path.join(main_file_path, 'Validation', '*.mat')))


hf = h5py.File(os.path.join(save_path, Train_Data_name), 'w-')
data_set = hf.create_dataset("data", (1, 51, 51, 3), maxshape=(None, 51, 51, 3), dtype='float32')
label_set = hf.create_dataset("labels", (1, 1), maxshape=(None, 1), dtype='float32')
itr = 0
for Train_n in range(0, len(Train_files)):
    file_path = Train_files[Train_n]
    if Train_n % 1000 == 0:
        print(str(Train_n) + ':' + file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = data
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.fliplr(data)
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.flipud(data)
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.rot90(data)
    label_set[itr, :] = labels
    itr = itr + 1

hf.close()

hf = h5py.File(os.path.join(save_path, Valid_Data_name), 'w-')
data_set = hf.create_dataset("data", (1, 51, 51, 3), maxshape=(None, 51, 51, 3), dtype='float32')
label_set = hf.create_dataset("labels", (1, 1), maxshape=(None, 1), dtype='float32')
# noinspection PyRedeclaration
itr = 0
for Valid_n in range(0, len(Valid_files)):
    file_path = Valid_files[Valid_n]
    if Valid_n % 1000 == 0:
        print(str(Valid_n) + ':' + file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = data
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.fliplr(data)
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.flipud(data)
    label_set[itr, :] = labels
    itr = itr + 1
    data_set.resize(((itr + 1), 51, 51, 3))
    label_set.resize(((itr + 1), 1))
    data_set[itr, :, :, :] = np.rot90(data)
    label_set[itr, :] = labels
    itr = itr + 1

hf.close()





