# Some of the datasets are too large to be uploaded to github.
# If you need the datasets, please email yiyang.sun@duke.edu
import numpy as np 

def data_prep(data):
    data_path = '../data/'
    if data == "MNIST":
        X = np.load(data_path + '/mnist_images.npy', allow_pickle=True).reshape(70000, 28 * 28)
        labels = np.load(data_path + '/mnist_labels.npy', allow_pickle=True)
    elif data == "FMNIST":
        X = np.load(data_path + '/fmnist_images.npy', allow_pickle=True).reshape(70000, 28 * 28)
        labels = np.load(data_path + '/fmnist_labels.npy', allow_pickle=True)
    elif data == "USPS":
        X = np.load(data_path + '/USPS.npy', allow_pickle=True)
        labels = np.load(data_path + '/USPS_labels.npy', allow_pickle=True)
    elif data == "20NG":
        X = np.load(data_path + '/20NG.npy', allow_pickle=True)
        labels = np.load(data_path + '/20NG_labels.npy', allow_pickle=True)
    elif data == "COIL20":
        X = np.load(data_path + '/coil_20.npy', allow_pickle=True).reshape(1440, 128 * 128)
        labels = np.load(data_path + '/coil_20_labels.npy', allow_pickle=True)
    elif data == "kang":
        X = np.load(data_path + "/kang_log_pca.npy", allow_pickle=True)
        labels = np.load(data_path + "/kang_labels.npy", allow_pickle=True)
    elif data =="CBMC":
        X = np.load(data_path + "/CBMC.npy", allow_pickle=True)
        labels = np.load(data_path + "/CBMC_labels.npy", allow_pickle=True)
    elif data == "human_cortex":
        X = np.load(data_path + "/human_cortex.npy", allow_pickle=True)
        labels = np.load(data_path + "/human_cortex_labels.npy", allow_pickle=True)
    elif data == "seurat":
        X = np.load(data_path + "/seurat_data.npy", allow_pickle=True)
        labels = np.load(data_path + "/seurat_label.npy", allow_pickle=True)
    return X, labels