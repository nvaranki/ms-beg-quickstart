# !/usr/bin/env python
# coding: utf-8

# This is a MindSpore (华为/Huawei) sample Python script.
# conda install mindspore-cpu=1.6.1 -c mindspore -c conda-forge
# https://anaconda.org/MindSpore/mindspore-cpu/files mindspore-cpu-1.7.0-py39_0.tar.bz2
# conda install "mindspore-cpu-1.7.0-py39_0.tar.bz2"

import os
import requests
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import context
from mindspore.ops import operations as P
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor
import skimage.io
# from IPython.display import Image
# from IPython.display import display


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Mul(nn.Cell):

    # Simple code to verify MindScore installation

    def __init__(self):
        super(Mul, self).__init__()
        self.mul = P.Mul()

    def construct(self, x, y):
        return self.mul(x, y)


x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
mul = Mul()
print(mindspore.__version__)
print(mul(x, y))

# MindScore example code:


def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1, shuffle=False):
    # Define the dataset.
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # Define the mapping to be operated.
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # Use the map function to apply data operations to the dataset.
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op], input_columns="image",
                            num_parallel_workers=num_parallel_workers)

    # Perform shuffle, batch and repeat operations.
    buffer_size = 10000
    #     if shuffle :
    #         mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(count=repeat_size)

    return mnist_ds


class LeNet5(nn.Cell):
    """
    LeNet network structure
    """

    def __init__(self, num_class=10, num_channel=1):
        """ Memory and Error
        400*120+120=48120 + 120*84+84=10164 + 84*10+10=850 = 59134  fc1+fc2+fc3  1.0%
        400*400+400=160400 + 400*10+10=4010 = 164410  fc1+fc3  0.9%
        400*10+10=4010  fc3  1.2%
        """
        super(LeNet5, self).__init__()
        # Define the required operation.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 400, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(400, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # Original
    # epoch: 1 step: 1875, loss is 0.41728726029396057
    # {'Accuracy': 0.9659455128205128}
    # Success: 9629/10000 96%, Failure: 371/10000  4%
    # ----------------------
    # fc1++, no fc2
    # epoch: 1 step: 1875, loss is 0.06656097620725632
    # {'Accuracy': 0.9707532051282052}
    # Success: 9769/10000 98%, Failure: 231/10000  2%

    def construct(self, x):
        # Use the defined operation to construct a forward network.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define a training method."""
    # Load the training dataset.
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size, shuffle=True)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


def test_net(model, data_path):
    """Define a validation method."""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


def test_net_ex(model, file_path):
    # Load the saved model for testing.
    param_dict = load_checkpoint(file_path)
    # Load parameters to the network.
    load_param_into_net(model.predict_network, param_dict) #

    # Define a test dataset. If batch_size is set to 1, an image is obtained.
    ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1)

    if os.path.exists("images"):
        for fn in os.listdir("images"):
            if fn.endswith(".png"):
                os.remove("images/"+fn)

    success = 0
    failed = 0
    i = 0
    for data in ds_test.create_dict_iterator():

        # `images` indicates the test image, and `labels` indicates the actual classification of the test image.
        images = data["image"].asnumpy()
        labels = data["label"].asnumpy()

        # Use the model.predict function to predict the classification of the image.
        output = model.predict(Tensor(data['image'])).asnumpy()[0]
        # print(output)

        predicted = np.argmax(output)
        actual = labels[0]

        # Output the predicted classification and the actual classification.
        if predicted == actual:
            success += 1
        else:
            failed += 1
            print(f'{i:5d} Predicted: {predicted} @{output[predicted]:5.3f}, Actual: {actual} @{output[actual]:5.3f}')
            bmp = data["image"].asnumpy()[0, 0,]
            save_image(i, bmp, f'p{predicted}a{actual}')
        i += 1

    total = success + failed
    print(f'Success: {success}/{total} {success / total:5.1%}, Failure: {failed}/{total} {failed / total:5.1%}')


def save_image(i, bmp, suffix="test"):
    # print(bmp)
    h = bmp.shape[1]
    w = bmp.shape[0]
    img = np.zeros((w, h, 3), dtype="uint8")
    for y in range(h):
        for x in range(w):
            g = int(np.floor(bmp[x, y] * 255))
            img[x, y,] = [g, g, g]
    # print(img)
    fname = "images"
    try:
        if not os.path.exists(fname):
            os.makedirs(fname)
        fname += f"/{i:05d}-{suffix}.png"
        if not os.path.exists(fname):
            skimage.io.imsave(fname, img, check_contrast=False)
    except Exception:
        print(f"Failed to save \"{fname}\".")


if __name__ == '__main__':

    train_path = "datasets/MNIST_Data/train"
    test_path = "datasets/MNIST_Data/test"
    train_epoch = 5
    mnist_path = "./datasets/MNIST_Data"
    dataset_size = 1

    # Press the green button in the gutter to run the script.

    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    args = parser.parse_known_args()[0]
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte",train_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte",train_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte",test_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte",test_path)

    # Instantiate the network.
    net = LeNet5()

    # Define the loss function.
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Define the optimizer.
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # Set model saving parameters.
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # Use model saving parameters.
    ckpoint = ModelCheckpoint(directory="checkpoints", prefix="checkpoint_lenet", config=config_ck)

    # Train and test the net:
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
    test_net(model, mnist_path)
    test_net_ex(model, ckpoint.latest_ckpt_file_name)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

