import os
import sys
import re
import math
import time
import random
import requests
import hashlib
import collections
import zipfile
import tarfile
import shutil
import mindspore
import numpy as np
import pandas as pd
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import Tensor, Parameter
from matplotlib import pyplot as plt

d2l = sys.modules[__name__]

from IPython import display
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.dataset as ds
from mindspore.common.initializer import initializer, XavierUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
DATA_HUB['banana-detection'] = (DATA_URL + 'banana-detection.zip',
                                    '5de26c8fce5ccdea9f91267273464dc968d20d72')
DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')



class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class FashionMnist():
    def __init__(self, path, kind):
        self.data, self.label = load_mnist(path, kind)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class ArrayData():
    def __init__(self, data):
        assert len(data) > 1
        self.data = data

    def __getitem__(self, index):
        return (i[index] for i in self.data)

    def __len__(self):
        return len(self.data[0])


class SGD(nn.Cell):
    def __init__(self, lr, batch_size, parameters):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.parameters = parameters

    def construct(self, grads):
        for idx in range(len(self.parameters)):
            ops.assign(self.parameters[idx], self.parameters[idx] - self.lr * grads[idx] / self.batch_size)
        return True


class Train(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class NetWithLoss(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss


class TrainCh8(nn.Cell):
    def __init__(self, network, optimizer, grad_op):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_op = grad_op

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        grads = self.grad_op(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class NetWithLossCh8(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss


@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size)
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0])
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)


class AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.AvgPool(kernel_size, kernel_size)(x)


class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)


class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)


def try_gpu():
    return None


def linreg(x, w, b):
    return ops.matmul(x, w) + b


def squared_loss(y_hat, y):
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return images, labels


def load_data_fashion_mnist(batch_size, resize=None, works=1):
    """将Fashion-MNIST数据集加载到内存中。"""
    data_path = "../data"
    mnist_train = FashionMnist(data_path, kind='train')
    mnist_test = FashionMnist(data_path, kind='t10k')

    mnist_train = ds.GeneratorDataset(source=mnist_train, column_names=['image', 'label'], shuffle=False)
    mnist_test = ds.GeneratorDataset(source=mnist_test, column_names=['image', 'label'], shuffle=False)
    trans = [vision.Rescale(1.0 / 255.0, 0), vision.HWC2CHW()]
    type_cast_op = transforms.TypeCast(mindspore.int32)
    if resize:
        trans.insert(0, vision.Resize(resize))
    mnist_train = mnist_train.map(trans, input_columns=["image"])
    mnist_test = mnist_test.map(trans, input_columns=["image"])
    mnist_train = mnist_train.map(type_cast_op, input_columns=['label'])
    mnist_test = mnist_test.map(type_cast_op, input_columns=['label'])
    mnist_train = mnist_train.batch(batch_size, num_parallel_workers=works)
    mnist_test = mnist_test.batch(batch_size, num_parallel_workers=works)
    return mnist_train, mnist_test


def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.
    Defined in :numref:`sec_utils`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Mindspore data iterator.
    Defined in :numref:`sec_utils`"""
    dataset = ArrayData(data_arrays)
    data_column_size = len(data_arrays)
    dataset = ds.GeneratorDataset(source=dataset, column_names=[str(i) for i in range(data_column_size)],
                                  shuffle=is_train)
    type_cast_op = transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(type_cast_op, input_columns=str(data_column_size - 1))
    dataset = dataset.batch(batch_size)
    return dataset


def get_dataloader_workers():
    """Use 4 processes to read the data.
    Defined in :numref:`sec_fashion_mnist`"""
    return 4


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.
    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.asnumpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():
    """Use 4 processes to read the data.
    
    Defined in :numref:`sec_fashion_mnist`"""
    return 4


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.
    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.matmul(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X.astype(np.float32), y.reshape((-1, 1)).astype(np.float32)


def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.asnumpy() == y.asnumpy()
    return float(cmp.sum())


def evaluate_accuracy(net, dataset):
    """计算在指定数据集上模型的精度。"""
    metric = Accumulator(2)
    for X, y in dataset.create_tuple_iterator():
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]


def train_epoch_ch3(net, dataset, loss, optim):
    """训练模型一个迭代周期（定义见第3章）。"""

    # 定义前向网络
    def forward_fn(x, y):
        y_hat = net(x)
        l = loss(y_hat, y)
        return l

    batch_size = dataset.get_batch_size()
    metric = Accumulator(3)
    for X, y in dataset:
        grad_fn = mindspore.value_and_grad(forward_fn, grad_position=None, weights=optim.parameters)
        l, grads = grad_fn(X, y)
        y_hat = net(X)
        optim(grads)
        metric.add(float(l.asnumpy()), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2] * batch_size, metric[1] / metric[2]


def train_ch3(net, train_dataset, test_dataset, loss, num_epochs, optim):
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_dataset, loss, optim)
        test_acc = evaluate_accuracy(net, test_dataset)
        print(train_metrics)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics


def predict_ch3(net, dataset, n=6):
    """预测标签（定义见第3章）。"""
    for X, y in dataset.create_tuple_iterator():
        break
    trues = get_fashion_mnist_labels(y.asnumpy())
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_model_selection`"""
    if isinstance(net, nn.Cell):
        net.set_train(False)
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), l.size)
    return metric[0] / metric[1]


def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = mnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def evaluate_accuracy_gpu(net, dataset, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    net.set_train(False)
    metric = Accumulator(2)
    for X, y in dataset.create_tuple_iterator():
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]


def train_ch6(net, train_dataset, test_dataset, num_epochs, lr):
    """用GPU训练模型(在第六章定义)。"""
    init_cells(net=net)

    optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义前向传播函数
    def forward_fn(x, y):
        y_hat = net(x)
        loss = loss_fn(y_hat, y)
        return loss, y_hat

    grad_fn = ops.value_and_grad(forward_fn, None, weights=net.trainable_params(), has_aux=True)

    # 定义模型单步训练
    def train(X, Y, optim):
        (loss, y_hat), grads = grad_fn(X, Y)
        loss = ops.depend(loss, optim(grads))
        return loss, y_hat

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), train_dataset.get_dataset_size()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.set_train()
        for i, (X, y) in enumerate(train_dataset.create_tuple_iterator()):
            timer.start()
            loss, y_hat = train(X, y, optim)
            metric.add(loss.asnumpy() * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_dataset)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')


def evaluate_accuracy_gpu_bert(net, dataset, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    net.set_train(False)
    metric = Accumulator(2)
    for X1, X2, X3, y in dataset.create_tuple_iterator():
        metric.add(accuracy(net((X1, X2, X3)), y), y.size)
    return metric[0] / metric[1]


def train_ch15_bert(net, train_iter, test_iter, loss, trainer, num_epochs):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), train_iter.get_dataset_size()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])

    def forward_fn(inputs, targets):
        logits = net(inputs)
        l = loss(logits, targets)
        return l, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, trainer.parameters, has_aux=True)

    def train_step(inputs, targets):
        (l, logits), grads = grad_fn(inputs, targets)
        trainer(grads)
        return l, logits

    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        net.set_train()
        for i, (features1, features2, features3, labels) in enumerate(train_iter.create_tuple_iterator()):
            timer.start()
            l, acc = train_batch_ch13(
                train_step, (features1, features2, features3), labels)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(mindspore.get_context("device_target"))}')


def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感"""
    sequence = Tensor(vocab[sequence.split()], mindspore.int32)
    label = ops.argmax(net(sequence.reshape((1, -1))), axis=1)
    return 'positive' if label == 1 else 'negative'


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名。
    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件。
    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩。'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件。
    Defined in :numref:`sec_kaggle_house`"""
    for name in DATA_HUB:
        download(name)


def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率。"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列。"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield mindspore.Tensor(X, mindspore.int32), mindspore.Tensor(Y, mindspore.int32)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = mindspore.Tensor(corpus[offset: offset + num_tokens], mindspore.int32)
    Ys = mindspore.Tensor(corpus[offset + 1: offset + 1 + num_tokens], mindspore.int32)
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def predict_ch8(prefix, num_preds, net, vocab):
    """在`prefix`后面生成新字符。"""
    net.set_train(False)
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(Tensor([outputs[-1]], mindspore.int32), (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1).asnumpy()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


class TrainCh8(nn.Cell):
    def __init__(self, network, optimizer, theta):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
        self.theta = theta

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        grads = ops.clip_by_global_norm(grads, self.theta)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class NetWithLossCh8(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss


def grad_clipping(grads, theta):
    """裁剪梯度。"""
    norm = ops.sqrt(sum(ops.sum((g ** 2)) for g in grads))
    if norm > theta:
        for g in grads:
            g[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和，词元数量

    # 定义前向函数
    def forward_fn(x, state, y):
        y_hat, state = net(x, state)
        l = loss(y_hat, y).mean()
        return l

    # 获取梯度函数
    grad_fn = mindspore.value_and_grad(forward_fn, None, weights=net.trainable_params())
    net.set_train()
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0])
        y = Y.T.reshape(-1)
        (l), grads = grad_fn(X, state, y)
        grad_clipping(grads, 1)
        if isinstance(updater, nn.Optimizer):
            updater(grads)
        else:
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l.asnumpy() * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Cell):
        updater = nn.SGD(net.trainable_params(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒')
    print(predict('time traveller'))
    print(predict('traveller'))


class RNNModel(nn.Cell):
    """循环神经网络模型。"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Dense(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Dense(self.num_hiddens * 2, self.vocab_size)

    def construct(self, inputs, state):
        X = ops.one_hot(inputs.T, self.vocab_size, d2l.tensor(1.0), d2l.tensor(0.0))
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return ops.zeros((self.num_directions * self.rnn.num_layers,
                              batch_size, self.num_hiddens))
        else:
            # nn.LSTM以元组作为隐状态
            return (ops.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens)),
                    ops.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens)))


class RNNModelScratch(nn.Cell):
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        super().__init__()
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def construct(self, X, state):
        X = ops.one_hot(X.T, self.vocab_size, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)


class Encoder(nn.Cell):
    """编码器-解码器架构的基本编码器接口"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def construct(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Cell):
    """编码器-解码器架构的基本解码器接口"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def construct(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Cell):
    """编码器-解码器架构的基类"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def preprocess_nmt(text):
    """预处理“英语－法语”数据集。"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines], dtype=np.int32)
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], np.int32), 1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.
    Defined in :numref:`sec_attention-cues`"""
    use_svg_display()
    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.asnumpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0., **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = d2l.Embedding(vocab_size, embed_size, embedding_table='normal')
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def construct(self, X, X_len=None):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.transpose(1, 0, 2)
        output, state = self.rnn(X)
        return output, state


class MaskedSoftmaxCELoss(nn.Cell):
    """带遮蔽的softmax交叉熵损失函数"""

    def __init__(self):
        super().__init__()
        self.softmax_ce_loss = nn.CrossEntropyLoss()

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def construct(self, pred, label, valid_len):
        weights = ops.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        unweighted_loss = self.softmax_ce_loss(pred.transpose(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(axis=1)
        return weighted_loss


class NetWithLossCh8_Seq2seq(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-2])
        loss = self.loss(y_hat, inputs[-2], inputs[-1])
        return loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device=None):
    """训练序列到序列模型"""

    optimizer = nn.Adam(net.trainable_params(), lr)
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])

    def forward_fn(X, dec_input, X_valid_len, Y, Y_valid_len):
        pred, _ = net(X, dec_input, X_valid_len)
        l = loss(pred, Y, Y_valid_len)
        return l

    grad_fn = mindspore.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=False)

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        net.set_train()
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.astype(d2l.int32) for x in batch]
            # print(X.shape, X_valid_len, Y.shape, Y_valid_len)
            bos = mindspore.Tensor([tgt_vocab['<bos>']] * Y.shape[0], dtype=mindspore.int32).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # 强制教学
            l, grads = grad_fn(X, dec_input, X_valid_len, Y, Y_valid_len)
            optimizer(grads)
            num_tokens = Y_valid_len.sum()
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec')


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.set_train(False)
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = mindspore.Tensor([len(src_tokens)], mindspore.int32)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = d2l.expand_dims(mindspore.Tensor(src_tokens, mindspore.int32), 0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = d2l.expand_dims(mindspore.Tensor([tgt_vocab['<bos>']], mindspore.int32), 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(axis=2)
        pred = int(dec_X.squeeze(0).asnumpy())
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    maxlen = X.shape[1]
    mask = ops.arange((maxlen), dtype=mindspore.float32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作"""
    if valid_lens is None:
        return nn.Softmax(-1)(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = d2l.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.Softmax(-1)(X.reshape(shape))


def init_cells(net):
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape))
            if cell.has_bias:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(initializer(Uniform(bound), [cell.out_channels]))


class AdditiveAttention(nn.Cell):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Dense(key_size, num_hiddens, has_bias=False)
        self.W_q = nn.Dense(query_size, num_hiddens, has_bias=False)
        self.w_v = nn.Dense(num_hiddens, 1, has_bias=False)
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = d2l.expand_dims(queries, 2) + d2l.expand_dims(keys, 1)
        features = d2l.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return d2l.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Cell):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = d2l.bmm(queries, keys.swapaxes(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return d2l.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状。"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.transpose(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 函数的操作。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Cell):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, has_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = Dense(query_size, num_hiddens, has_bias=has_bias)
        self.W_k = Dense(key_size, num_hiddens, has_bias=has_bias)
        self.W_v = Dense(value_size, num_hiddens, has_bias=has_bias)
        self.W_o = Dense(num_hiddens, num_hiddens, has_bias=has_bias)

    def construct(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = d2l.repeat(
                valid_lens, repeats=self.num_heads, axis=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Cell):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(1 - dropout)
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=mindspore.float32).reshape(
            -1, 1) / d2l.pow(10000, d2l.arange(
            0, num_hiddens, 2, dtype=mindspore.float32) / num_hiddens)
        self.P[:, :, 0::2] = d2l.sin(X)
        self.P[:, :, 1::2] = d2l.cos(X)

    def construct(self, X):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class PositionWiseFFN(nn.Cell):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = d2l.Dense(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = d2l.Dense(ffn_num_hiddens, ffn_num_outputs)

    def construct(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Cell):
    """残差连接后进行层规范化。"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(1 - dropout)
        self.ln = d2l.LayerNorm(normalized_shape)

    def construct(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Cell):
    """transformer编码器块。"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def construct(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = d2l.Embedding(vocab_size, num_hiddens, embedding_table='normal')
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.SequentialCell()
        for i in range(num_layers):
            self.blks.append(EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                          norm_shape, ffn_num_input, ffn_num_hiddens,
                                          num_heads, dropout, use_bias))

    def construct(self, X, valid_lens):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, gamma_init='ones', beta_init='zeros', epsilon=1e-05):
        """Initialize LayerNorm."""
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(f"For '{self.cls_name}', the type of 'normalized_shape' should be tuple[int] or list[int], "
                            f"but got {normalized_shape} and the type is {type(normalized_shape)}.")
        self.normalized_shape = normalized_shape
        self.norm_ndim = len(normalized_shape)
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")

    def construct(self, input_x):
        layer_norm = ops.LayerNorm(begin_norm_axis=input_x.ndim - self.norm_ndim,
                                   begin_params_axis=input_x.ndim - self.norm_ndim,
                                   epsilon=self.epsilon)
        y, _, _ = layer_norm(input_x, self.gamma, self.beta)
        return y


class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='xavier_uniform', bias_init='zeros', has_bias=has_bias,
                         activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding


def train_transformer(net, data_iter, lr, num_epochs, tgt_vocab):
    """训练序列到序列模型"""

    optimizer = nn.Adam(net.trainable_params(), lr)
    loss = MaskedSoftmaxCELoss()
    net_with_loss = NetWithLossCh8_Seq2seq(net, loss)
    train = TrainCh8(net_with_loss, optimizer, 1)
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.astype(mindspore.int32) for x in batch]
            bos = mindspore.Tensor([tgt_vocab['<bos>']] * Y.shape[0], mindspore.int32).reshape(-1, 1)
            dec_input = mnp.concatenate([bos, Y[:, :-1]], 1)
            batch_size, num_steps = X.shape
            dec_valid_lens = mnp.tile(mnp.arange(1, num_steps + 1), (batch_size, 1))
            l = train(X, dec_input, X_valid_len, dec_valid_lens, Y, Y_valid_len)
            num_tokens = Y_valid_len.sum()
            metric.add(l.sum().asnumpy(), num_tokens.asnumpy())
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec')


def predict_transformer(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
    """序列到序列模型的预测"""
    net.set_train(False)
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = mindspore.Tensor([len(src_tokens)], mindspore.int32)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = mnp.expand_dims(mindspore.Tensor(src_tokens, mindspore.int32), 0)
    enc_outputs, encoder_attention_weight = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = mnp.expand_dims(mindspore.Tensor([tgt_vocab['<bos>']], mindspore.int32), 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state, attention_weights = net.decoder(dec_X, dec_state, None)
        dec_X = Y.argmax(axis=2)
        pred = int(dec_X.squeeze(0).asnumpy())
        if save_attention_weights:
            attention_weight_seq.append(attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq, encoder_attention_weight


def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`subsec_gd-learningrate`"""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def train_batch_ch13(train_step, X, y):
    """用多GPU进行小批量训练"""
    l, pred = train_step(X, y)
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), train_iter.get_dataset_size()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])

    def forward_fn(inputs, targets):
        logits = net(inputs)
        l = loss(logits, targets)
        return l, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, trainer.parameters, has_aux=True)

    def train_step(inputs, targets):
        (l, logits), grads = grad_fn(inputs, targets)
        trainer(grads)
        return l, logits

    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        net.set_train()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                train_step, features, labels)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(mindspore.get_context("device_target"))}')


def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization.

    Defined in :numref:`subsec_gd-learningrate`"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')


def read_ptb():
    """将PTB数据集加载到文本行的列表中"""
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)


def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(*data):
    """返回带有负采样的跳元模型的小批量样本"""
    _data = data
    if isinstance(data[0], list):
        data = zip(data[0], data[1], data[2])
        _data = zip(_data[0], _data[1], _data[2])
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in _data:
        if not isinstance(center, (int, list)):
            center, context, negative = center.tolist(), context.tolist(), negative.tolist()
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers).reshape((-1, 1)), np.array(contexts_negatives),
            np.array(masks, dtype=np.float32), np.array(labels))


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下载PTB数据集，然后将其加载到内存中"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset:
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    data = PTBDataset(all_centers, all_contexts, all_negatives)
    dataset = mindspore.dataset.GeneratorDataset(data, column_names=["center", "context", "negative"], shuffle=True,
                                                 num_parallel_workers=num_workers)
    dataset = dataset.batch(batch_size=512, per_batch_map=batchify,
                            output_columns=['centers', 'contexts_negatives', 'masks', 'labels'])

    return dataset, vocab


def get_data_ch11(batch_size=10, n=1500):
    """Defined in :numref:`sec_minibatches`"""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下载PTB数据集，然后将其加载到内存中"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset:
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    data = PTBDataset(all_centers, all_contexts, all_negatives)
    dataset = mindspore.dataset.GeneratorDataset(data, column_names=["center", "context", "negative"], shuffle=True,
                                                 num_parallel_workers=num_workers)
    dataset = dataset.batch(batch_size=512, per_batch_map=batchify,
                            output_columns=['centers', 'contexts_negatives', 'masks', 'labels'])

    return dataset, vocab


def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    """Defined in :numref:`sec_minibatches`"""
    # Initialization
    w = mindspore.Parameter(d2l.normal(mean=0.0, stddev=0.01, shape=(feature_dim, 1)))
    b = mindspore.Parameter(d2l.zeros((1)))
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    loss_fn = lambda x, y: loss(net(x), y).mean()
    grad_fn = mindspore.value_and_grad(loss_fn, None, [w, b])
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l, [dw, db] = grad_fn(X, y)
            trainer_fn([w, b], [dw, db], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / data_iter.get_dataset_size(),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    """Defined in :numref:`sec_minibatches`"""
    # Initialization
    net = nn.SequentialCell(nn.Dense(5, 1))

    def init_weights(m):
        if type(m) == nn.Dense:
            m.weight.set_data(initializer(Normal(0.01), m.weight.shape))
    
    net.apply(init_weights)
    optimizer = trainer_fn(net.trainable_params(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    forward_fn = lambda X, y: loss(net(X), y).mean()

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, net.trainable_params())
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            output, grads = grad_fn(X, y)
            optimizer(grads)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss计算平方误差时不带系数1/2
                animator.add(n / X.shape[0] / data_iter.get_dataset_size(),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')


d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')


class TokenEmbedding:
    """GloVe嵌入"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        # fastText网站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, mindspore.Tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[mindspore.Tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')


def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签"""

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
    if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                  in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataset:
    """用于加载SNLI数据集的自定义数据集"""

    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                                   all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = dataset[2]
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return [d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
            for line in lines]

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
    transform_label = transforms.TypeCast(mindspore.int32)
    num_parallel_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = ds.GeneratorDataset(train_set, shuffle=True, column_names=['data', 'label'],
                                     num_parallel_workers=num_parallel_workers)
    train_iter = train_iter.map(input_columns="label", operations=transform_label)
    train_iter = train_iter.batch(batch_size=batch_size)
    test_iter = ds.GeneratorDataset(test_set, shuffle=False, column_names=['data', 'label'],
                                    num_parallel_workers=num_parallel_workers)
    test_iter = test_iter.map(input_columns="label", operations=transform_label)
    test_iter = test_iter.batch(batch_size=batch_size)

    return train_iter, test_iter, train_set.vocab


d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')


def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels.

    Defined in :numref:`sec_sentiment`"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset.

    Defined in :numref:`sec_sentiment`"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = [d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens]
    test_features = [d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens]
    train_iter = load_array((train_features, train_data[1]), batch_size)
    test_iter = load_array((test_features, test_data[1]),
                           batch_size,
                           is_train=False)
    return train_iter, test_iter, vocab


class BERTEncoder(nn.Cell):
    """BERT编码器"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.SequentialCell()
        for i in range(num_layers):
            self.blks.append(d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = mindspore.Parameter(ops.randn(1, max_len, num_hiddens),
                                                 name="pos_embedding")

    def construct(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Cell):
    """BERT的掩蔽语言模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.SequentialCell(nn.Dense(num_inputs, num_hiddens),
                                     nn.ReLU(),
                                     nn.LayerNorm(tuple([num_hiddens])),
                                     nn.Dense(num_hiddens, vocab_size))

    def construct(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = ops.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = ops.repeat_interleave(batch_idx, num_pred_positions, 0)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Cell):
    """BERT的下一句预测任务"""

    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(num_inputs, 2)

    def construct(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


class BERTModel(nn.Cell):
    """BERT模型"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.SequentialCell(nn.Dense(hid_in_features, num_hiddens),
                                        nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def construct(self, tokens, segments, valid_lens=None,
                  pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
                max_len - len(token_ids)), dtype=np.int64))
        all_segments.append(np.array(segments + [0] * (
                max_len - len(segments)), dtype=np.int64))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(np.array(len(token_ids), dtype=np.float32))
        all_pred_positions.append(np.array(pred_positions + [0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=np.int64))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                    max_num_mlm_preds - len(pred_positions)),
                     dtype=np.float32))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=np.int64))
        nsp_labels.append(np.array(is_next, dtype=np.int64))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset:
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                     + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_dataset = ds.GeneratorDataset(train_set, column_names=["all_token_ids", "all_segments",
                                                                 "valid_lens", "all_pred_positions", "all_mlm_weights",
                                                                 "all_mlm_labels", "nsp_labels"], shuffle=True,
                                        num_parallel_workers=num_workers)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset, train_set.vocab


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    # mlm_Y_hat = mlm_Y_hat.astype("")
    mlm_Y = mlm_Y.astype("int32")
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * \
            mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_y = nsp_y.astype("int32")
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_dataset, net, loss, vocab_size, num_steps):
    trainer = nn.Adam(net.trainable_params(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False

    def forward_fn(vocab_size, tokens_X, segments_X, valid_lens_x,
                   pred_positions_X, mlm_weights_X, mlm_Y, nsp_y):
        mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                                               segments_X, valid_lens_x,
                                               pred_positions_X, mlm_weights_X,
                                               mlm_Y, nsp_y)
        return mlm_l, nsp_l, l

    grad_fn = mindspore.value_and_grad(forward_fn, None, weights=net.trainable_params(), has_aux=True)
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
                mlm_weights_X, mlm_Y, nsp_y in train_dataset.create_tuple_iterator():
            (mlm_l, nsp_l, l), grads = grad_fn(
                vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            trainer(grads)
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on ')

def get_params(img, size):
    """
    Get parameters for crop for a random crop.

    Args:
        img (numpy.ndarray): Image to be cropped.
        size (tuple): Expected output size of the crop.

    Returns:
        params (i, j, h, w) to be passed to crop for random crop.
    """
    img_height, img_width = img.shape[:2]
    height, width = size
    if height > img_height or width > img_width:
        raise ValueError("Crop size {} is larger than input image size {}.".format(size, (img_height, img_width)))

    if width == img_width and height == img_height:
        return 0, 0, img_height, img_width

    top = random.randint(0, img_height - height)
    left = random.randint(0, img_width - width)
    return top, left, height, width

def crop(img, top, left, height, width):
    """
    Crop the input numpy.ndarray.

    Args:
        img (numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image,
            in the directions of (width, height).
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        numpy.ndarray, cropped image.
    """

    return img[top : top + height, left : left + width ]

class Residual(nn.Cell):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides, pad_mode='pad', weight_init='xavier_uniform')
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, pad_mode='pad', weight_init='xavier_uniform')
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides, weight_init='xavier_uniform')
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def construct(self, X):
        Y = ops.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return ops.relu(Y)

def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model.

    Defined in :numref:`sec_multi_gpu_concise`"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.SequentialCell(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer

    net = nn.SequentialCell(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init='xavier_uniform'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        resnet_block(64, 64, 2, first_block=True),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2),
        resnet_block(256, 512, 2),
        d2l.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dense(512, num_classes, weight_init='xavier_uniform')
    )

    return net


def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    num_sizes, num_ratios = len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = mindspore.Tensor(sizes)
    ratio_tensor = mindspore.Tensor(ratios)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (ops.arange(in_height) + offset_h) * steps_h
    center_w = (ops.arange(in_width) + offset_w) * steps_w
    shift_y, shift_x = ops.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = ops.concat((size_tensor * ops.sqrt(ratio_tensor[0]),
                   sizes[0] * ops.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = ops.concat((size_tensor / ops.sqrt(ratio_tensor[0]),
                   sizes[0] / ops.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = ops.tile(ops.stack((-w, -h, w, h)).T,
                                    (in_height * in_width, 1)) / 2
    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = ops.stack([shift_x, shift_y, shift_x, shift_y],
                axis=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.

    Defined in :numref:`sec_bbox`"""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = ops.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = ops.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = ops.full((num_anchors,), -1, dtype=mindspore.int64)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices  = ops.max(jaccard, axis=1)
    anc_i = ops.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = ops.masked_select(indices, max_ious >= iou_threshold)
    anchors_bbox_map[anc_i] = box_j
    col_discard = ops.full((num_anchors,), -1)
    row_discard = ops.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = ops.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * ops.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = ops.concat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors)
        bbox_mask = ops.tile((anchors_bbox_map >= 0).float().unsqueeze(-1), (1, 4))
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = ops.zeros(num_anchors, dtype=mindspore.int64)
        assigned_bb = ops.zeros((num_anchors, 4), dtype=mindspore.float32)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = ops.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[:, 0][bb_idx].long() + 1
        assigned_bb[indices_true] = label[:, 1:][bb_idx]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = ops.stack(batch_offset)
    bbox_mask = ops.stack(batch_mask)
    class_labels = ops.stack(batch_class_labels).astype('int32')
    return (bbox_offset, bbox_mask, class_labels)

d2l.DATA_HUB['banana-detection'] = (d2l.DATA_URL + 'banana-detection.zip',
                                    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(ds.vision.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, mindspore.Tensor(targets, dtype=mindspore.float32).unsqueeze(1) / 255

class BananasDataset():
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.parent = None
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (np.array(self.features[int(idx)], dtype='float32'), self.labels[int(idx)])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = ds.GeneratorDataset(source=BananasDataset(is_train=True),
                                     column_names=['imgs', 'labels'], shuffle=True)
    val_iter = ds.GeneratorDataset(source=BananasDataset(is_train=False),
                                     column_names=['imgs', 'labels'], shuffle=False)
    train_iter = train_iter.map(mindspore.dataset.vision.HWC2CHW(),input_columns='imgs')
    train_iter = train_iter.batch(batch_size=batch_size, drop_remainder=True)
    val_iter = val_iter.map(mindspore.dataset.vision.HWC2CHW(),input_columns='imgs')
    val_iter = val_iter.batch(batch_size=batch_size, drop_remainder=True)
    return train_iter, val_iter


def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = ops.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = ops.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = ops.argsort(scores, axis=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i.asnumpy())
        if B.numel() == 1: break
        iou = box_iou(boxes[i].reshape(-1, 4),
                      boxes[:, :][B[1:]].reshape(-1, 4)).reshape(-1)
        inds = ops.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return mindspore.Tensor(keep)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    batch_size = cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = ops.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = ops.arange(num_anchors, dtype=mindspore.int64)
        combined = ops.concat((keep, all_idx))
        unique = ops.unique(combined)
        uniques, _ = unique[0], unique[1]
        # 统计未重复的，mindspore的unique无法统计同一个元素出现次数
        counts = mnp.bincount(combined)
        non_keep = mindspore.Tensor([int(uniques[i]) for i in range(len(counts))
                                     if (counts[i] == 1)])
        all_id_sorted = ops.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id = mindspore.Tensor.float(class_id)
        class_id[below_min_idx] = -1
        for j in range(len(conf)):
            if below_min_idx[j] == True:
                conf[j] = 1 - conf[j]

        pred_info = ops.concat((class_id.unsqueeze(1),
                                conf.unsqueeze(1).astype('float32'),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return ops.stack(out)

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = mindspore.dataset.vision.ImageReadMode.COLOR
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(mindspore.dataset.vision.read_image(os.path.join( # read as numpy.ndarray
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(mindspore.dataset.vision.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = np.zeros(256 ** 3, dtype=np.int64)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = d2l.get_params(feature, (height, width))
    feature = d2l.crop(feature, *rect)
    label = d2l.crop(label, *rect)
    return feature, label

class VOCSegDataset():
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = mindspore.dataset.vision.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.astype('float32') / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = mindspore.dataset.GeneratorDataset(VOCSegDataset(True, crop_size, voc_dir), column_names=["data", "label"],
                                                    shuffle=True, num_parallel_workers =num_workers)
    train_iter = train_iter.map(mindspore.dataset.vision.HWC2CHW())
    train_iter = train_iter.batch(batch_size=batch_size, drop_remainder=True)
    test_iter = mindspore.dataset.GeneratorDataset(VOCSegDataset(False, crop_size, voc_dir), column_names=["data", "label"],
                                                   shuffle=True, num_parallel_workers =num_workers)
    test_iter = test_iter.map(mindspore.dataset.vision.HWC2CHW())
    test_iter = test_iter.batch(batch_size=batch_size, drop_remainder=True)
    return train_iter, test_iter

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

def copyfile(filename, target_dir):
    """Copy a file into a target directory.

    Defined in :numref:`sec_kaggle_cifar10`"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
    
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary.

    Defined in :numref:`sec_kaggle_cifar10`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set.

    Defined in :numref:`sec_kaggle_cifar10`"""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction.

    Defined in :numref:`sec_kaggle_cifar10`"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
class RoiPooling():
    """for fast R-CNN"""
    def __init__(self, mode='tf', pool_size=(7,7)):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        :param mode:
        :param pool_size:
        """
        self.mode = mode
        self.pool_size = pool_size

    def pool(self, region):
        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = np.zeros((region_channels, pool_height, pool_width))
        h_step = int(region_height / pool_height)
        w_step = int(region_width / pool_width)
        for i in range(pool_height):
            for j in range(pool_width):

                xmin = j * w_step
                xmax = xmin + 1
                ymin = i * h_step
                ymax = ymin + 1

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                if xmin==xmax or ymin==ymax:
                    continue
                if self.mode=='tf':
                    pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                elif self.mode=='th':
                    region_new = region[:, ymin:ymax+1, xmin:xmax+1]
                    pool[:, i, j] = np.max(np.array(region_new), axis=(1,2))

        return pool

    def get_region(self, feature_map, roi_dimensions):
        """
        fetching the roi from feature map by the dimension of the roi
        :param feature_map: the feature map with size of (1, height, width, channels)
        :param roi_dimensions: a region of interest dimensions
        :return:
        """
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if self.mode=='tf':
            feature_map = np.squeeze(feature_map)
            if feature_map.shape == 2:
                feature_map = feature_map.unsqueeze(0)
            r = feature_map[ymin:ymax, xmin:xmax, :]
        elif self.mode=='th':
            feature_map = np.squeeze(feature_map)
            if len(feature_map.shape) == 2:
                feature_map = feature_map.unsqueeze(0)
            r = feature_map[:, ymin:ymax+1, xmin:xmax+1]
        return r

    def get_pooled_rois(self,feature_map, roi_batch):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        pool = []
        for region_dim in roi_batch:
            region = self.get_region(feature_map, region_dim)
            p = self.pool(region)
            pool.append(p)
        return np.array(pool)
    

abs = ops.abs
arange = ops.arange
randn = ops.randn
concat = ops.concat
flatten = ops.flatten
int32 = mindspore.int32
float32 = mindspore.float32
ones = ops.ones
zeros = ops.zeros
inner = ops.inner
mv = ops.mv
mm = ops.mm
matmul = ops.matmul
stack = ops.stack
sin = ops.sin
cos = ops.cos
sinh = ops.sinh
cosh = ops.cosh
tanh = ops.tanh
exp = ops.exp
square = ops.square
sqrt = ops.sqrt
sign = ops.sign
softmax = ops.softmax
meshgrid = ops.meshgrid
linspace = ops.linspace
zeros_like = ops.zeros_like
log = ops.log
tensor_dot = ops.tensor_dot
maximum = ops.maximum
relu = ops.relu
sigmoid = ops.sigmoid
norm = ops.norm
cat = ops.cat
sort = ops.sort
expand_dims = ops.expand_dims
tile = ops.tile
eye = lambda n: ops.eye(n, dtype=float32)
numpy = lambda x, *args, **kwargs: x.asnumpy(*args, **kwargs)
pow = lambda x, y: ops.pow(x, y)
clip_by_value = lambda x, clip_value_min, clip_value_max: ops.clip_by_value(x, clip_value_min, clip_value_max)
uniform = lambda shape, minval, maxval: ops.uniform(shape, tensor(minval, float32), tensor(maxval, float32),
                                                    dtype=float32)
rand = lambda size, *args: ops.rand(size, dtype=float32)
randn = lambda size, *args: ops.randn(size, dtype=float32)
tensor = lambda x, *args, **kwargs: mindspore.Tensor(x, *args, **kwargs)
normal = lambda shape, mean, stddev, *args: ops.normal(shape, tensor(mean), tensor(stddev), *args)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.astype(*args, **kwargs)
repeat = lambda x, repeats, *args, **kwargs: x.repeat(repeats, *args, **kwargs)
bmm = lambda x, y: ops.bmm(x, y)
