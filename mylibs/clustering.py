from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix

def label_data(unlabeled_data, num_classes):
    print("Labeling unlabeled data...")
    offset = 10 - num_classes
    images = np.array([x[0].numpy() for x in unlabeled_data])
    actual_labels = np.array([x[1] for x in unlabeled_data])
    # pca_2 = PCA(n_components=2)
    # pca_2_result = pca_2.fit_transform(images)
    # print(pca_2_result.shape)

    # kmeans clustering the images
    kmeans = KMeans(n_clusters=num_classes).fit(images.reshape(images.shape[0], -1))
    predict_labels = kmeans.labels_ + offset

    # relabel to match the original labels to be consistent with test dataset
    rearrange_labels = np.copy(predict_labels)
    cmatrix = confusion_matrix(predict_labels, actual_labels)
    for i in range(num_classes):
        mask = predict_labels == i + offset
        rearrange_labels[mask] = np.argmax(cmatrix[i]) + offset
        
    accuracy = np.sum(rearrange_labels == actual_labels) / len(actual_labels)
    print("Labeling accuracy: {}".format(accuracy))
    return unlabeled_data