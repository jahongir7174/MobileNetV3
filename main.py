import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
from timm import utils
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = os.path.join('..', 'Dataset', 'IMAGENET')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def lr(args):
    return 0.256 * args.batch_size * args.world_size / 4096


def mix_up(samples, targets, model, criterion):
    alpha = numpy.random.beta(1.0, 1.0)
    index = torch.randperm(samples.size()[0]).cuda()

    samples = alpha * samples + (1 - alpha) * samples[index, :]

    with torch.cuda.amp.autocast():
        outputs = model(samples)
    return criterion(outputs, targets) * alpha + criterion(outputs, targets[index]) * (1 - alpha)


def cut_mix(samples, targets, model, criterion):
    shape = samples.size()
    index = torch.randperm(shape[0]).cuda()
    alpha = numpy.sqrt(1. - numpy.random.beta(1.0, 1.0))

    w = numpy.int(shape[2] * alpha)
    h = numpy.int(shape[3] * alpha)

    # uniform
    c_x = numpy.random.randint(shape[2])
    c_y = numpy.random.randint(shape[3])

    x1 = numpy.clip(c_x - w // 2, 0, shape[2])
    y1 = numpy.clip(c_y - h // 2, 0, shape[3])
    x2 = numpy.clip(c_x + w // 2, 0, shape[2])
    y2 = numpy.clip(c_y + h // 2, 0, shape[3])

    samples[:, :, x1:x2, y1:y2] = samples[index, :, x1:x2, y1:y2]

    alpha = 1 - ((x2 - x1) * (y2 - y1) / (shape[-1] * shape[-2]))

    with torch.cuda.amp.autocast():
        outputs = model(samples)
    return criterion(outputs, targets) * alpha + criterion(outputs, targets[index]) * (1. - alpha)


def train(args):
    util.setup_seed()
    util.setup_multi_processes()

    model = nn.MobileNetV3().cuda()
    ema_m = nn.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = nn.RMSprop(util.params(model), lr(args))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])

    scheduler = nn.StepLR(optimizer)
    criterion = nn.CrossEntropyLoss().cuda()

    sampler = None
    dataset = Dataset(os.path.join(data_dir, 'train'),
                      transforms.Compose([util.Resize(size=args.input_size),
                                          util.RandomAugment(mean=9.0, n=2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize]))
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                             sampler=sampler, num_workers=8, pin_memory=True)

    with open('weights/step.csv', 'w') as f:
        best = 0

        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch',
                                                   'acc@1', 'acc@5',
                                                   'train_loss', 'val_loss'])
            writer.writeheader()
        for epoch in range(args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            p_bar = loader
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(loader, total=len(loader))
            model.train()
            m_loss = util.AverageMeter()
            for samples, targets in p_bar:
                samples = samples.cuda()
                targets = targets.cuda()

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()

                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                torch.cuda.synchronize()
                ema_m.update(args, model)

                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                m_loss.update(loss.item(), samples.size(0))

                if args.local_rank == 0:
                    gpus = '%.4gG' % (torch.cuda.memory_reserved() / 1E9)
                    desc = ('%10s' * 2 + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), gpus, m_loss.avg)
                    p_bar.set_description(desc)

            scheduler.step(epoch + 1)

            if args.local_rank == 0:
                last = test(args, ema_m.model)
                writer.writerow({'acc@1': str(f'{last[1]:.3f}'),
                                 'acc@5': str(f'{last[2]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'val_loss': str(f'{last[0]:.3f}'),
                                 'train_loss': str(f'{m_loss.avg:.3f}')})
                f.flush()

                state = {'model': copy.deepcopy(ema_m.model)}
                torch.save(state, 'weights/last.pt')
                if last[1] > best:
                    torch.save(state, 'weights/best.pt')

                del state

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    if model is None:
        model = torch.load('weights/best.pt', 'cuda')['model'].float().fuse()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    dataset = Dataset(os.path.join(data_dir, 'val'),
                      transforms.Compose([transforms.Resize(args.input_size + 32),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 32, num_workers=8, pin_memory=True)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    m_loss = util.AverageMeter()

    for samples, targets in tqdm.tqdm(loader, ('%10s' * 3) % ('acc@1', 'acc@5', 'loss')):
        samples = samples.cuda()
        targets = targets.cuda()

        with torch.cuda.amp.autocast():
            outputs = model(samples)

        torch.cuda.synchronize()

        acc1, acc5 = util.accuracy(outputs, targets, (1, 5))

        top1.update(acc1.item(), samples.size(0))
        top5.update(acc5.item(), samples.size(0))
        m_loss.update(criterion(outputs, targets).item(), samples.size(0))

    acc1, acc5 = top1.avg, top5.avg
    print('%10.3g' * 3 % (acc1, acc5, m_loss.avg))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return m_loss.avg, acc1, acc5


def profile(args):
    model = nn.MobileNetV3().export().eval()
    shape = (1, 3, args.input_size, args.input_size)

    model(torch.zeros(shape))
    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')
    if args.benchmark:
        util.print_benchmark(model, shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
