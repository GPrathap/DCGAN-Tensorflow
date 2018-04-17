from __future__ import print_function

import glob
import os
import sys
import tarfile
import subprocess
import argparse
import h5py
import numpy as np
import scipy.misc

parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+',
                    choices=['MNIST', 'Fashion', 'SVHN', 'CIFAR10'])

def prepare_h5py(train_image, test_image, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)

    print ('Preprocessing data...')

    import progressbar
    from time import sleep
    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    data_id = open(os.path.join(data_dir,'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i%(image.shape[0]/100)==0:
            bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
    bar.finish()
    f.close()
    data_id.close()
    return

def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'mnist')
    if os.path.exists(data_dir):
        print('MNIST was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test,28,28,1)).astype(np.float)

    prepare_h5py(train_image, test_image, data_dir)

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)


def download_fashion_mnist(download_path):
    data_dir = os.path.join(download_path, 'fashion_mnist')
    if os.path.exists(data_dir):
        print('Fashion MNIST was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    data_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    keys = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test,28,28,1)).astype(np.float)

    prepare_h5py(train_image, test_image, data_dir)

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)


def download_svhn(download_path):

    import scipy.io as sio
    # svhn file loader
    def svhn_loader(url, path):
        cmd = ['curl', url, '-o', path]
        subprocess.call(cmd)
        m = sio.loadmat(path)
        return np.transpose(m['X'], (3, 0, 1, 2))

    data_dir = os.path.join(download_path, 'svhn')
    if os.path.exists(data_dir):
        print('SVHN was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    print('Downloading SVHN')
    data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    train_image = svhn_loader(data_url, os.path.join(data_dir, 'train_32x32.mat'))

    data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    test_image = svhn_loader(data_url, os.path.join(data_dir, 'test_32x32.mat'))

    prepare_h5py(train_image, test_image,  data_dir)

    cmd = ['rm', '-f', os.path.join(data_dir, '*.mat')]
    subprocess.call(cmd)

def createFiles(path, classes):
    datalist = {}
    datalist["images"] = []
    datalist["labels"] = []
    for filename in glob.iglob(path, recursive=True):

        image = scipy.misc.imread(filename).astype(np.uint8)
        # image = tf.gfile.FastGFile(filename, 'rb').read()
        currentClass = ""
        for cla in classes:
            if cla in filename:
                currentClass = cla
                break
        if currentClass == "":
            print("NO claass is foune rggreub")
        else:
            datalist["images"].append(image.flatten())
            datalist["labels"].append(classes.index(currentClass))
    return datalist

def download_cifar10(download_path):

    data_dir = '/data/cifar10'

    classesList = ["agricultural", 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral'
        , 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential',
                   'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                   'storagetanks', 'tenniscourt']
    datapath_traing = "/data/images/uc_train_256_data/**.jpg"
    datalist_traing = createFiles(datapath_traing, classesList)

    datapath_testing = "/data/images/uc_test_256/**.jpg"
    datalist_testing = createFiles(datapath_testing, classesList)

    traing_images = np.array(datalist_traing['images'])
    testing_images = np.array(datalist_testing['images'])

    prepare_h5py(traing_images, testing_images, data_dir, [256, 256, 3])


if __name__ == '__main__':
    args = parser.parse_args()
    path = './datasets'
    if not os.path.exists(path): os.mkdir(path)

    if 'MNIST' in args.datasets:
        download_mnist('./datasets')
    if 'Fashion' in args.datasets:
        download_fashion_mnist('./datasets')
    if 'SVHN' in args.datasets:
        download_svhn('./datasets')
    if 'CIFAR10' in args.datasets:
        download_cifar10('./datasets')
