import math
import random
from os import environ
from platform import system

import cv2
import numpy
import torch
from PIL import Image, ImageOps, ImageEnhance

max_value = 10.0


def print_benchmark(model, shape):
    import os
    import onnx
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx.backend import Caffe2Backend
    from caffe2.python import core, model_helper, workspace

    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    # export torch to onnx
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, './weights/model.onnx', True, False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    onnx.checker.check_model(onnx.load('./weights/model.onnx'))

    # export onnx to caffe2
    onnx_model = onnx.load('./weights/model.onnx')

    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)

    # print benchmark
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    init_net_proto.ParseFromString(caffe2_init.SerializeToString())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    predict_net_proto.ParseFromString(caffe2_predict.SerializeToString())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape, mean=0.0, std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    # remove onnx model
    os.remove('./weights/model.onnx')


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch + 1)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def params(model, decay=1e-5):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.}, {'params': p2, 'weight_decay': decay}]


def accuracy(output, target, top_k):
    with torch.no_grad():
        output = output.topk(max(top_k), 1, True, True)[1].t()
        output = output.eq(target.view(1, -1).expand_as(output))

        results = []
        for k in top_k:
            correct = output[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct.mul_(100.0 / target.size(0)))
        return results


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def cutout(image, m):
    w, h = image.size

    min_area = 0.02
    max_area = 1 / 3 * m / max_value

    min_ratio = math.log(0.3)
    max_ratio = math.log(1 / 0.3)

    for _ in range(10):
        area = random.uniform(min_area, max_area) * h * w
        ratio = math.exp(random.uniform(min_ratio, max_ratio))

        cut_w = int(round(math.sqrt(area / ratio)))
        cut_h = int(round(math.sqrt(area * ratio)))

        if cut_w < w and cut_h < h:
            y = random.randint(0, h - cut_h)
            x = random.randint(0, w - cut_w)

            image = numpy.array(image)
            image[y:y + cut_h, x:x + cut_w] = 0
            image = Image.fromarray(image)

            break
    return image


def rotate(image, m):
    m = (m / max_value) * 60.0

    if random.random() < 0.5:
        m *= -1

    return image.rotate(m, resample=resample())


def shear_x(image, m):
    m = (m / max_value) * 0.30

    if random.random() < 0.5:
        m *= -1

    return image.transform(image.size, Image.AFFINE, (1, m, 0, 0, 1, 0), resample=resample())


def shear_y(image, m):
    m = (m / max_value) * 0.30

    if random.random() < 0.5:
        m *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, m, 1, 0), resample=resample())


def translate_x(image, m):
    m = (m / max_value) * 0.50

    if random.random() < 0.5:
        m *= -1

    pixels = m * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, m):
    m = (m / max_value) * 0.50

    if random.random() < 0.5:
        m *= -1

    pixels = m * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def identity(image, _):
    return image


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, m):
    if random.random() < 0.5:
        m = (m / max_value) * 1.8 + 0.1
    else:
        m = (m / max_value) * 0.9

        if random.random() < 0.5:
            m = max(0.1, 1.0 - m)
        else:
            m = max(0.1, 1.0 + m)

    return ImageEnhance.Brightness(image).enhance(m)


def color(image, m):
    if random.random() < 0.5:
        m = (m / max_value) * 1.8 + 0.1
    else:
        m = (m / max_value) * 0.9

        if random.random() < 0.5:
            m = max(0.1, 1.0 - m)
        else:
            m = max(0.1, 1.0 + m)

    return ImageEnhance.Color(image).enhance(m)


def contrast(image, m):
    if random.random() < 0.5:
        m = (m / max_value) * 1.8 + 0.1
    else:
        m = (m / max_value) * 0.9

        if random.random() < 0.5:
            m = max(0.1, 1.0 - m)
        else:
            m = max(0.1, 1.0 + m)

    return ImageEnhance.Contrast(image).enhance(m)


def sharpness(image, m):
    if random.random() < 0.5:
        m = (m / max_value) * 1.8 + 0.1
    else:
        m = (m / max_value) * 0.9

        if random.random() < 0.5:
            m = max(0.1, 1.0 - m)
        else:
            m = max(0.1, 1.0 + m)

    return ImageEnhance.Sharpness(image).enhance(m)


def solar(image, m):
    if random.random() < 0.5:
        m = int((m / max_value) * 256)
        if random.random() < 0.5:
            m = 256 - m
    else:
        m = int((m / max_value) * 110)
    return ImageOps.solarize(image, m)


def poster(image, m):
    m = int((m / max_value) * 4)
    if random.random() < 0.5:
        if m >= 8:
            return image
    else:
        if random.random() < 0.5:
            m = max(4 - m, 1)
        else:
            m = min(4 + m, 8)

        if m >= 8:
            return image
    return ImageOps.posterize(image, m)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([size, size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class RandomAugment:
    def __init__(self, mean: float = 9.0, sigma: float = 1.0, n: int = 2):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          brightness, color, contrast, sharpness, solar, poster,
                          cutout, rotate, shear_x, shear_y, translate_x, translate_y)

    def __call__(self, image):
        for transform in numpy.random.choice(self.transform, self.n):
            m = numpy.random.normal(self.mean, self.sigma)
            m = min(max_value, max(0.0, m))

            image = transform(image, m)
        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
