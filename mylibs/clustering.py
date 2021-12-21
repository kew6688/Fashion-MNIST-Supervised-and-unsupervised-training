from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data.dataloader import DataLoader

from mylibs.loss import autoencoder_loss
from mylibs.train import autoencoder_train
from mylibs.model import Autoencoder

EPOCH = 15

def encode(images, all_data, USE_GPU=False):
    print("Training Auto Encoder...")
    device = torch.device("cuda" if USE_GPU else "cpu")
    model = Autoencoder(input_height=28).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_dataloader = DataLoader(all_data, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(1, EPOCH+1):
        loss = autoencoder_train(train_dataloader, model, autoencoder_loss, optimizer, USE_GPU)
        print("Epoch: {} Loss: {}".format(epoch, loss))

    encode_dataloader = DataLoader(images, batch_size=1, shuffle=True, num_workers=2)
    return model.encode(encode_dataloader, USE_GPU)
    

# mode:
# 0 - kmeans
# 1 - kmeans with PCA
# 2 - kmeans with Auto Encoder
# 3 - Gaussian Mixture
# 4 - Gaussian Mixture with PCA
# 5 - Gaussian Mixture with Auto Encoder
def label_data(unlabeled_data, labels, all_data, mode=0, USE_GPU=False):
    print("Labeling unlabeled data...")
    num_classes = len(labels)
    images = np.array([x[0].numpy() for x in unlabeled_data])
    actual_labels = np.array([x[1] for x in unlabeled_data])

    if mode == 0 or mode == 3:
        images = images.reshape(len(unlabeled_data), -1)
    elif mode == 1 or mode == 4:
        # PCA to reduce dimensions of images
        pca = PCA(n_components=5)
        images = pca.fit_transform(images.reshape(len(unlabeled_data), -1))
    elif mode == 2 or mode == 5:
        # Auto Encoder to reduce dimensions of images
        images = encode(images, all_data, USE_GPU)

    if mode == 0 or mode == 1 or mode == 2:
        # kmeans clustering the images
        model = KMeans(n_clusters=num_classes)
    elif mode == 3 or mode == 4 or mode == 5:
        # Gaussian Mixture clustering the images
        model = GaussianMixture(n_components=num_classes, covariance_type='spherical')

    predict_labels = labels[model.fit_predict(images)]

    # relabel to match the original labels to be consistent with test dataset
    rearrange_labels = np.copy(predict_labels)
    cmatrix = confusion_matrix(predict_labels, actual_labels, labels=labels)
    for i in range(num_classes):
        mask = predict_labels == labels[i]
        rearrange_labels[mask] = labels[np.argmax(cmatrix[i])]

    accuracy = np.sum(rearrange_labels == actual_labels) / len(actual_labels)
    print("Labeling accuracy: {}".format(accuracy))
    
    for i in range(len(unlabeled_data)):
        unlabeled_data[i] = (unlabeled_data[i][0], rearrange_labels[i])
    return unlabeled_data