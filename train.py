import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import VideoDataset
import slowfast
from tensorboardX import SummaryWriter
from utils import evaluate


def total_right(output, target):
    pred = output.argmax(dim=1)
    return sum(pred == target)


def train_batch(model, data_loader, optimizer, scheduler, criterion):
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input, target = [d.to(device) for d in data_loader]
    target = target.type(torch.LongTensor).to(device)
    output = model(input)

    batch_size = input.size(0)
    pred = output.argmax(dim=1)

    right_predict = total_right(output, target)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    scheduler.step()
    return loss.item(), right_predict.item()


def train(model, train_loader, val_loader,  epoch, criterion, scheduler, optimizer, writer, load_checkpoint):
    # switch to train mode
    model.train()
    end = time.time()

    # load checkpoint config
    step = load_checkpoint['step']
    best_acc = load_checkpoint['best_acc']
    current_epoch = load_checkpoint['epoch'] + 1


    for epoch in range(current_epoch, current_epoch + epoch):
        print(f'epoch: {epoch}')
        tot_train_loss = 0
        tot_train_count = 0
        correct = 0

        for train_data in train_loader:
            loss, right_predict = train_batch(model, train_data, optimizer, scheduler, criterion)
            train_size = train_data[0].size(0)

            # update config
            tot_train_loss += loss
            correct += right_predict
            tot_train_count += train_size

            if step % params['display'] == 0:
                print(f'train_batch_loss[{step}]: ', loss / train_size, "correct", right_predict / train_size)
                writer.add_scalar('training loss', loss / train_size, step // params['display'])
                writer.add_scalar('training acc', right_predict / train_size, step // params['display'])

            if step % params['validate'] == 0:
                evaluation = evaluate(model, val_loader, criterion)
                val_loss, val_acc = evaluation['loss'], evaluation['acc']
                print(f'valid_evaluation: loss {val_loss}, acc {val_acc}')
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_model_path = f'checkpoint/checkpoint_{epoch}.pt'
                    torch.save({
                        'best_acc': best_acc,
                        'epoch': epoch,
                        'step': step,
                        'state_dict': model.state_dict()},
                        save_model_path
                    )
                    print('best_acc', best_acc)
                    print('save model at', save_model_path)
                writer.add_scalar('valid loss', val_loss, step // params['validate'])
                writer.add_scalar('valid acc', val_acc, step // params['validate'])

            step += 1

            print("train loss", tot_train_loss / tot_train_count)
            print("train accu", correct / tot_train_count)
            print()


def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    train_loader = DataLoader(VideoDataset(params['dataset'], mode='train', clip_len=params['clip_len'],
                                               frame_sample_rate=params['frame_sample_rate']),
                                  batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_loader = DataLoader(VideoDataset(params['dataset'], mode='validation', clip_len=params['clip_len'],
                                             frame_sample_rate=params['frame_sample_rate']),
                                batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    print("load model")
    model = slowfast.resnet50(class_num=params['num_classes'])

    if params['checkpoint'] is not None:
        load_checkpoint = torch.load(params['checkpoint'])
        pretrained_dict = load_checkpoint['state_dict']

        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=params['betas'],
                           weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)

    epochs = params['epoch_num']
    train(model, train_loader, val_loader, epochs, criterion, scheduler, optimizer, writer, load_checkpoint)

    writer.close


if __name__ == '__main__':
    main()
