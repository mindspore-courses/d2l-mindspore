import time
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from matplotlib import pyplot as plt
from IPython import display
import mindspore.dataset.vision.c_transforms as cv
import mindspore.dataset as ds

def load_data_fashion_mnist(data_path, batch_size, resize=None, works=1):  
    """将Fashion-MNIST数据集加载到内存中。"""
    mnist_train = ds.MnistDataset(data_path, usage='train')
    mnist_test = ds.MnistDataset(data_path, usage='test')
    if resize:
        trans = [cv.Resize(resize)]
        mnist_train = mnist_train.map(trans, input_columns=["image"])
        mnist_test = mnist_test.map(trans, input_columns=["image"])
    mnist_train = mnist_train.batch(batch_size, num_parallel_workers=works)
    mnist_test = mnist_test.batch(batch_size, num_parallel_workers=works)
    return mnist_train, mnist_test

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.
    Defined in :numref:`sec_utils`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

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
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
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
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def synthetic_data(w, b, num_examples):  
    """生成 y = Xw + b + 噪声。"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.matmul(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X.astype(np.float32), y.reshape((-1, 1)).astype(np.float32)

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

class SGD(nn.Cell):
    def __init__(self, lr, batch_size, params):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.params = params
        
    def construct(self, grads):
        for idx in range(len(self.params)):
            ops.assign(self.params[idx], self.params[idx] - self.lr * grads[idx] / self.batch_size)
        return True
    
class Train(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
        
    def construct(self, x, y):
        loss = self.network(x, y)
        grads = self.grad(self.network, self.optimizer.params)(x, y)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class NetWithLoss(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss
        
    def construct(self, x, y):
        y_hat = self.network(x)
        loss = self.loss(y_hat, y)
        return loss