import math
import time
import mindspore
import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import Tensor
from matplotlib import pyplot as plt
from IPython import display
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset as ds

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
    def __init__(self, data, label):
        self.data, self.label = data, label
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)

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
        
    def construct(self, x, y):
        loss = self.network(x, y)
        grads = self.grad(self.network, self.optimizer.parameters)(x, y)
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

def load_data_fashion_mnist(data_path, batch_size, resize=None, works=1):  
    """将Fashion-MNIST数据集加载到内存中。"""
    mnist_train = FashionMnist(data_path, kind='train')
    mnist_test = FashionMnist(data_path, kind='t10k')

    mnist_train = ds.GeneratorDataset(source=mnist_train, column_names=['image', 'label'], shuffle=False)
    mnist_test = ds.GeneratorDataset(source=mnist_test, column_names=['image', 'label'], shuffle=False)
    trans = [CV.Rescale(1.0 / 255.0, 0), CV.HWC2CHW()]
    type_cast_op = C.TypeCast(mindspore.int32)
    if resize:
        trans.insert(0, CV.Resize(resize))
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
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_utils`"""
    dataset = ArrayData(data_arrays[0].astype(np.float32), data_arrays[1].astype(np.float32))
    dataset = ds.GeneratorDataset(source=dataset, column_names=['data', 'label'], shuffle=is_train)
    dataset = dataset.batch(batch_size)
    return dataset

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
    net_with_loss = nn.WithLossCell(net, loss)
    net_train = nn.TrainOneStepCell(net_with_loss, optim)
    metric = Accumulator(3)
    batch_size = dataset.get_batch_size()

    for X, y in dataset.create_tuple_iterator():
        l = net_train(X, y)
        y_hat = net(X)
        metric.add(float(l.asnumpy()), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2] * batch_size, metric[1] / metric[2]

def train_ch3(net, train_dataset, test_dataset, loss, num_epochs, optim):  
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_dataset, loss, optim)
        print(train_metrics)
        test_acc = evaluate_accuracy(net, test_dataset)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

def predict_ch3(net, dataset, n=6):  
    """预测标签（定义见第3章）。"""
    for X, y in dataset.create_tuple_iterator():
        break
    trues = get_fashion_mnist_labels(y.asnumpy())
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
def evaluate_loss(net, dataset):
    """Evaluate the loss of a model on the given dataset.
    Defined in :numref:`sec_utils`"""
    net.set_train(False)
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in dataset.create_tuple_iterator():
        l = net(X, y)
        metric.add(l.sum().asnumpy(), l.size)
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
    optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_loss = nn.WithLossCell(net, loss)
    train = nn.TrainOneStepCell(net_with_loss, optim)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), train_dataset.get_dataset_size()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.set_train()
        for i, (X, y) in enumerate(train_dataset.create_tuple_iterator()):
            timer.start()
            l = train(X, y)
            y_hat = net(X)
            metric.add(l.asnumpy() * X.shape[0], accuracy(y_hat, y), X.shape[0])
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