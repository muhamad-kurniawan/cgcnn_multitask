import argparse
import os
import shutil
import sys
import time
import warnings
import json
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--transfer', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--config-alt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error

    # load data
    if args.config_alt:
      config_file = args.config_alt
    else:
      config_file = args.data_options[0] + '/config.json'
    with open(config_file) as f:
        config = json.load(f)
    dataset = CIFData(*args.data_options, config=config)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)
    print(f'len train_loader:{len(train_loader)}')
    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                output_nodes=config["nodes"],
                                # classification=True if args.task ==
                                #                        'classification' else False
                                tasks=config["tasks"]
                               )
  
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total_model_params:{total_params}')
  
    if args.cuda:
        model.cuda()

    # obtain target value normalizer
    criterions = []
    normalizers = []
    for idx, task in enumerate(config["tasks"]):
      # define loss func and optimizer
      if task == 'classification':
          criterions.append(nn.NLLLoss())
          normalizer = Normalizer(torch.zeros(2))
          normalizer.load_state_dict({'mean': 0., 'std': 1.})
          normalizers.append(normalizer)
          # normalizers.append(None)
      else:
          criterions.append(nn.MSELoss())
          if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
          else:
              sample_data_list = [dataset[i] for i in
                                  sample(range(len(dataset)), 500)]
          _, sample_target, _ = collate_pool(sample_data_list)
          normalizers.append(Normalizer(sample_target[idx]))
          # normalizers.append(normalizers_)
    
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    if torch.cuda.is_available():
      device = torch.device('cuda:0')
    else:
      device = torch.device('cpu')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizers = [norm.load_state_dict(checkpoint['normalizers'][i]) for i, norm in enumerate(normalizers)]
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.transfer:
        if os.path.isfile(args.transfer):
            print("=> loading checkpoint '{}'".format(args.transfer))
            checkpoint = torch.load(args.transfer, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # normalizers = [norma.load_state_dict(checkpoint['normalizers'][i]) for i, norma in enumerate(normalizers)]

            model_dict = model.state_dict()
            pretrained_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        pretrained_dict[k] = v
                    else:
                        print(f"Shape mismatch for layer {k}: "
                              f"pretrained shape {v.shape} vs model shape {model_dict[k].shape}")
                else:
                    print(f"Layer {k} not found in model")
        
            # print(f'dict_val:{list(pretrained_dict.values())}')
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
          
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.transfer, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.transfer))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        print(f'start epoch {epoch}')
        # train for one epoch
        get_embedding=False
        if epoch+1==args.epochs:
          get_embedding=True
          pass
        train(train_loader, model, criterions, optimizer, epoch, normalizers, config=config, get_embedding=get_embedding)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterions, normalizers, config=config, epoch=epoch, get_embedding=get_embedding)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizers': [norm.state_dict() for norm in normalizers],
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    # best_checkpoint = torch.load('model_best.pth.tar')
    # best_checkpoint = torch.load('checkpoint.pth.tar')
    # model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterions, normalizers, test=True, config=config)

def train(train_loader, model, criterions, optimizer, epoch, normalizers, config, get_embedding=False):
  tasks = config['tasks']
  if 'weights_loss' in config.keys():
    weights_loss = config['weights_loss']
  else:
    weights_loss = [1]*len(config['tasks'])
  batch_time = AverageMeter()
  data_time = AverageMeter()  
  scores = {}
  for t in range(len(tasks)):
    task_id = f'task_{t}'
    dict_task = {}
    # dict_task['batch_time'] = AverageMeter()
    # dict_task['data_time'] = AverageMeter()
    dict_task['losses'] = AverageMeter()
    if tasks[t] == 'regression':
        dict_task['mae_errors'] = AverageMeter()
    else:
        dict_task['accuracies'] = AverageMeter()
        dict_task['precisions'] = AverageMeter()
        dict_task['recalls'] = AverageMeter()
        dict_task['fscores'] = AverageMeter()
        dict_task['auc_scores'] = AverageMeter()
    scores[task_id] = dict_task
  embedding_dict = {}

    # switch to train mode
  model.train()

  end = time.time()
  print('start train_loader')
  # for i, (input, targets, _) in enumerate(train_loader):
  #   print(f'index i:{i}')
  for i, (input, targets, cif_id) in enumerate(train_loader):
    # print(f'index i:{i}')
    # for t_c in targets[2]:
    #   try:
        

    # measure data loading time
    data_time.update(time.time() - end)

    if args.cuda:
        input_var = (Variable(input[0].cuda(non_blocking=True)),
                     Variable(input[1].cuda(non_blocking=True)),
                     input[2].cuda(non_blocking=True),
                     [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
    else:
        input_var = (Variable(input[0]),
                     Variable(input[1]),
                     input[2],
                     input[3])
    # normalize target
    targets_var = []
    for idx, t in enumerate(tasks):
      if t == 'regression':
          target_normed = normalizers[idx].norm(targets[idx])
      else:
          target_normed = targets[idx].view(-1).long()
      if args.cuda:
          targets_var.append(Variable(target_normed.cuda(non_blocking=True)))
      else:
          targets_var.append(Variable(target_normed))

    # compute output
    error_target = False
    outputs, embedding = model(*input_var)
    losses = 0
    target_task_class = [t[2] for t in targets]
    # if os.path.exists("/content/target.txt"):
    #   os.remove("/content/target.txt")
    # with open('/content/target.txt', 'a') as f:
    #   f.write(str(targets))
    for idx, output in enumerate(outputs):

      task_id = f'task_{idx}'
      target = targets[idx]
      loss = criterions[idx](output, targets_var[idx])

      # measure accuracy and record loss
      if tasks[idx] == 'regression':
          mae_error = mae(normalizers[idx].denorm(output.data.cpu()), target)
          scores[task_id]['losses'].update(loss.data.cpu(), target.size(0))
          scores[task_id]['mae_errors'].update(mae_error, target.size(0))
      else:
          for n in target.numpy():
            try:
              int(n)
            except:
              print('class target is not int')
              error_target = True
          if error_target==False:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            scores[task_id]['losses'].update(loss.data.cpu().item(), target.size(0))
            scores[task_id]['accuracies'].update(accuracy, target.size(0))
            scores[task_id]['precisions'].update(precision, target.size(0))
            scores[task_id]['recalls'].update(recall, target.size(0))
            scores[task_id]['fscores'].update(fscore, target.size(0))
            scores[task_id]['auc_scores'].update(auc_score, target.size(0))
      losses += loss*weights_loss[idx]
      
    # compute gradient and do SGD step
    optimizer.zero_grad()
    if error_target==False:
      losses.backward()
      optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    embedding = embedding.cpu().numpy().tolist()
    if get_embedding:
      for idx_, cid in enumerate(cif_id):
        embedding_dict[cid] = embedding[idx_]

    if i % args.print_freq == 0:
      if error_target==False:
        for idx, task in enumerate(tasks):
          task_id = f'task_{idx}'
          if task == 'regression':
              print('Task_Id: {task_id}\t'
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                  epoch, i, len(train_loader), task_id=task_id, batch_time=batch_time,
                  data_time=data_time, loss=scores[task_id]['losses'], mae_errors=scores[task_id]['mae_errors'])
              )
          else:
              print('Task_Id: {task_id}\t'
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                    # 'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                    # 'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                    # 'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                    'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                  epoch, i, len(train_loader), task_id=task_id, batch_time=batch_time,
                  data_time=data_time, loss=scores[task_id]['losses'], accu=scores[task_id]['accuracies'],
                  prec=scores[task_id]['precisions'], recall=scores[task_id]['recalls'], f1=scores[task_id]['fscores'],
                  auc=scores[task_id]['auc_scores'])
              )
  if get_embedding:
    with open('embeddings.json', 'w') as f:
      json.dump(embedding_dict, f)

def validate(val_loader, model, criterions, normalizers, config, epoch=None, test=False, get_embedding=False):
  if 'weights_loss' in config.keys():
    weights_loss = config['weights_loss']
  else:
    weights_loss = [1]*len(config['tasks'])
  tasks = config['tasks']
  batch_time = AverageMeter()  
  scores = {}
  for t in range(len(tasks)):
    task_id = f'task_{t}'
    dict_task = {}
    # dict_task['batch_time'] = AverageMeter()
    # dict_task['data_time'] = AverageMeter()
    dict_task['losses'] = AverageMeter()
    if tasks[t] == 'regression':
        dict_task['mae_errors'] = AverageMeter()
    else:
        dict_task['accuracies'] = AverageMeter()
        dict_task['precisions'] = AverageMeter()
        dict_task['recalls'] = AverageMeter()
        dict_task['fscores'] = AverageMeter()
        dict_task['auc_scores'] = AverageMeter()
    if test:
        dict_task['test_targets'] = []
        dict_task['test_preds'] = []
        dict_task['test_cif_ids'] = []
      
    scores[task_id] = dict_task
    embedding_dict = {}
        
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # if args.task == 'regression':
    #     mae_errors = AverageMeter()
    # else:
    #     accuracies = AverageMeter()
    #     precisions = AverageMeter()
    #     recalls = AverageMeter()
    #     fscores = AverageMeter()
    #     auc_scores = AverageMeter()
        
    # switch to evaluate mode
  model.eval()

  end = time.time()
  for i, (input, targets, batch_cif_ids) in enumerate(val_loader):
      
      if args.cuda:
          with torch.no_grad():
              input_var = (Variable(input[0].cuda(non_blocking=True)),
                           Variable(input[1].cuda(non_blocking=True)),
                           input[2].cuda(non_blocking=True),
                           [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
      else:
          with torch.no_grad():
              input_var = (Variable(input[0]),
                           Variable(input[1]),
                           input[2],
                           input[3])

      targets_var = []
      # print(normalizers)
      # print(len(normalizers))
      for idx, t in enumerate(tasks):
        if t == 'regression':
            target_normed = normalizers[idx].norm(targets[idx])
        else:
            target_normed = targets[idx].view(-1).long()
        if args.cuda:
            targets_var.append(Variable(target_normed.cuda(non_blocking=True)))
        else:
            targets_var.append(Variable(target_normed))

      # if args.task == 'regression':
      #     target_normed = normalizer.norm(target)
      # else:
      #     target_normed = target.view(-1).long()
      # if args.cuda:
      #     with torch.no_grad():
      #         target_var = Variable(target_normed.cuda(non_blocking=True))
      # else:
      #     with torch.no_grad():
      #         target_var = Variable(target_normed)
      error_target = False
      # compute output
      outputs, embedding = model(*input_var)
      total_loss = 0
      for idx, output in enumerate(outputs):
        task_id = f'task_{idx}'
        target = targets[idx]
        loss = criterions[idx](output, targets_var[idx])
        total_loss += loss.item() * target.size(0)
        print(f'task_id: losses:{loss}')
        # loss = criterion(output, target_var)

        # measure accuracy and record loss

        if tasks[idx] == 'regression':
            mae_error = mae(normalizers[idx].denorm(output.data.cpu()), target)
            scores[task_id]['losses'].update(loss.data.cpu(), target.size(0))
            # scores[task_id]['mae_errors'].update(mae_error, target.size(0))
            if test:
                test_pred = normalizers[idx].denorm(output.data.cpu())
                test_target = target
                scores[task_id]['test_preds'] += test_pred.view(-1).tolist()
                scores[task_id]['test_targets'] += test_target.view(-1).tolist()
                scores[task_id]['test_cif_ids'] += batch_cif_ids
        else:
            for n in target.numpy():
              try:
                int(n)
              except:
                print('class target is not int')
              error_target = True
            if error_target == False:
              accuracy, precision, recall, fscore, auc_score = \
                  class_eval(output.data.cpu(), target)
              scores[task_id]['losses'].update(loss.data.cpu().item(), target.size(0))
              scores[task_id]['accuracies'].update(accuracy, target.size(0))
              scores[task_id]['precisions'].update(precision, target.size(0))
              scores[task_id]['recalls'].update(recall, target.size(0))
              scores[task_id]['fscores'].update(fscore, target.size(0))
              scores[task_id]['auc_scores'].update(auc_score, target.size(0))
              if test:
                  test_pred = torch.exp(output.data.cpu())
                  test_target = target
                  assert test_pred.shape[1] == 2
                  scores[task_id]['test_preds'] += test_pred[:, 1].tolist()
                  scores[task_id]['test_targets'] += test_target.view(-1).tolist()
                  scores[task_id]['test_cif_ids'] += batch_cif_ids

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
      embedding = embedding.cpu().numpy().tolist()
      if get_embedding or test==True:
        for idx_, cid in enumerate(batch_cif_ids):
          embedding_dict[cid] = embedding[idx_]


      if i % args.print_freq == 0:
        if error_target == False:
         for idx, task in enumerate(tasks):
            task_id = f'task_{idx}'
            if task == 'regression':
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time, loss=scores[task_id]['losses'],
                  mae_errors=scores[task_id]['mae_errors']))
              #         if args.task == 'regression':
              # print('Test: [{0}/{1}]\t'
              #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              #       'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
              #     i, len(val_loader), batch_time=batch_time, loss=scores[task_id]['losses'],
              #     mae_errors=scores[task_id]['mae_errors']))
            else:
                print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=scores[task_id]['losses'],
                    accu=scores[task_id]['accuracies'], prec=scores[task_id]['precisions'], recall=scores[task_id]['recalls'],
                    f1=scores[task_id]['fscores'], auc=scores[task_id]['auc_scores']))
              
  if get_embedding or test==True:
    if test== False:
      file_name = 'embeddings_val.json'
    else:
      file_name = 'embeddings_test.json'
    with open(file_name, 'w') as f:
      json.dump(embedding_dict, f)
      
  for idx, t in enumerate(tasks):
    task_id = f'task_{idx}'
    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(scores[task_id]['test_cif_ids'], scores[task_id]['test_targets'],
                                            scores[task_id]['test_preds']):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    # if t == 'regression':
    #     print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
    #                                                     mae_errors=scores[task_id]['mae_errors']))
    #     return scores[task_id]['mae_errors'].avg
      
    # else:
    #     print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
    #                                              auc=auc_scores[idx]))
    #     return scores[task_id]['auc_scores'].avg
  return total_loss


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
      
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    # if prediction.shape[1] == 2:
        # precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        #     target_label, pred_label, average='binary')
    print(f'target:{target_label[:5]}')
    print(f'pred:{pred_label[:5]}')
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label)
    # auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
    try:
      auc_score = metrics.roc_auc_score(target_label, prediction, multi_class='ovo')
    except:
      auc_score = 0.5
    accuracy = metrics.accuracy_score(target_label, pred_label)
    # else:
    #     raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
