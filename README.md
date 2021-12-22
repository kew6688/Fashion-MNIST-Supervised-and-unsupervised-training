# Fashion-MNIST-Supervised-and-unsupervised-training

"Adaptation to new classes": 

  assume that a classification network is trained to recognize K digits from 0 to K-1 or a subset of K object categories on Fashion MNIST dataset. For example, you can use full supervsion to train classification network on images only from these K categories. Then, add to your training database (unlabeled) images of N new categories (at least 2 more). That is, "eraze" the labels for all the images from the newly-added classes to simulated objects of unknown new classes. Design and implement unsupervised method to train or retrain a classification network to detect K+N classes assuming that the total number of distinct categories (K+N) is known. 
