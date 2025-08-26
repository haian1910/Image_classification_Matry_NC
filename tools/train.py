# import os
# import torch
# import torch.nn as nn
# import logging
# import time
# import random
# import numpy as np
# from torch.nn.parallel import DistributedDataParallel as DDP

# from lib.models.builder import build_model
# from lib.models.losses import CrossEntropyLabelSmooth, \
#     SoftTargetCrossEntropy
# from lib.dataset.builder import build_dataloader
# from lib.utils.optim import build_optimizer
# from lib.utils.scheduler import build_scheduler
# from lib.utils.args import parse_args
# from lib.utils.dist_utils import init_dist, init_logger
# from lib.utils.misc import accuracy, AverageMeter, \
#     CheckpointManager, AuxiliaryOutputBuffer
# from lib.utils.model_ema import ModelEMA
# from lib.utils.measure import get_params, get_flops

# try:
#     # need `pip install nvidia-ml-py3` to measure gpu stats
#     import nvidia_smi
#     nvidia_smi.nvmlInit()
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#     _has_nvidia_smi = True
# except ModuleNotFoundError:
#     _has_nvidia_smi = False


# torch.backends.cudnn.benchmark = True

# '''init logger'''
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S')
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


# def main():
#     args, args_text = parse_args()
#     args.exp_dir = f'experiments/{args.experiment}'

#     '''distributed'''
#     init_dist(args)
#     init_logger(args)

#     # save args
#     logger.info(args)
#     if args.rank == 0:
#         with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
#             f.write(args_text)

#     '''fix random seed'''
#     seed = args.seed + args.rank
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # torch.backends.cudnn.deterministic = True

#     '''build dataloader'''
#     train_dataset, val_dataset, train_loader, val_loader = \
#         build_dataloader(args)

#     '''build model'''
#     # Check if this is a Matryoshka model
#     is_matryoshka = args.model.startswith('matryoshka_')
    
#     if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
#         base_loss_fn = SoftTargetCrossEntropy()
#     elif args.smoothing == 0.:
#         base_loss_fn = nn.CrossEntropyLoss().cuda()
#     else:
#         base_loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
#                                               epsilon=args.smoothing).cuda()
    
#     # Initialize Matryoshka trainer if needed
#     matryoshka_trainer = None
#     if is_matryoshka:
#         from lib.utils.matryoshka_utils import MatryoshkaTrainer
#         from lib.models.losses import MatryoshkaLoss
        
#         # Use Matryoshka loss
#         loss_fn = MatryoshkaLoss(
#             matryoshka_dims=args.matryoshka_dims,
#             loss_weights=args.matryoshka_loss_weights,
#             base_loss_fn=base_loss_fn,
#             label_smoothing=args.smoothing
#         ).cuda()
        
#         # Initialize Matryoshka trainer
#         matryoshka_trainer = MatryoshkaTrainer(
#             matryoshka_dims=args.matryoshka_dims,
#             adaptive_training=args.matryoshka_adaptive,
#             warmup_epochs=args.matryoshka_warmup
#         )
#     else:
#         loss_fn = base_loss_fn
    
#     val_loss_fn = loss_fn

#     model = build_model(args, args.model)
#     logger.info(
#         f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
#         f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

#     # logger.info(
#     #     f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M')

#     # Diverse Branch Blocks
#     if args.dbb:
#         # convert 3x3 convs to dbb blocks
#         from lib.models.utils.dbb_converter import convert_to_dbb
#         convert_to_dbb(model)
#         logger.info(model)
#         logger.info(
#             f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
#             f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

#     model.cuda()
    

#     # knowledge distillation
#     if args.kd != '':
#         # build teacher model
#         teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
#         logger.info(
#             f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
#             f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
#         teacher_model.cuda()
#         test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
#         logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

#         # build kd loss
#         from lib.models.losses.kd_loss import KDLoss
#         loss_fn = KDLoss(model, teacher_model, args.model, args.teacher_model, loss_fn, 
#                          args.kd, args.ori_loss_weight, args.kd_loss_weight, args.kd_loss_kwargs)

#     model = DDP(model,
#                 device_ids=[args.local_rank],
#                 find_unused_parameters=False)
#     from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
#     model.register_comm_hook(None, comm_hooks.fp16_compress_hook)
#     if args.kd != '':
#         loss_fn.student = model
#     logger.info(model)

#     if args.model_ema:
#         model_ema = ModelEMA(model, decay=args.model_ema_decay)
#     else:
#         model_ema = None

#     '''build optimizer'''
#     optimizer = build_optimizer(args.opt,
#                                 model.module,
#                                 args.lr,
#                                 eps=args.opt_eps,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay,
#                                 filter_bias_and_bn=not args.opt_no_filter,
#                                 nesterov=not args.sgd_no_nesterov,
#                                 sort_params=args.dyrep)

#     '''build scheduler'''
#     steps_per_epoch = len(train_loader)
#     warmup_steps = args.warmup_epochs * steps_per_epoch
#     decay_steps = args.decay_epochs * steps_per_epoch
#     total_steps = args.epochs * steps_per_epoch
#     scheduler = build_scheduler(args.sched,
#                                 optimizer,
#                                 warmup_steps,
#                                 args.warmup_lr,
#                                 decay_steps,
#                                 args.decay_rate,
#                                 total_steps,
#                                 steps_per_epoch=steps_per_epoch,
#                                 decay_by_epoch=args.decay_by_epoch,
#                                 min_lr=args.min_lr)

#     '''dyrep'''
#     if args.dyrep:
#         from lib.models.utils.dyrep import DyRep
#         from lib.models.utils.recal_bn import recal_bn
#         dyrep = DyRep(
#             model.module,
#             optimizer,
#             recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
#                                            args.dyrep_recal_bn_iters, m),
#             filter_bias_and_bn=not args.opt_no_filter)
#         logger.info('Init DyRep done.')
#     else:
#         dyrep = None

#     '''amp'''
#     if args.amp:
#         loss_scaler = torch.cuda.amp.GradScaler()
#     else:
#         loss_scaler = None

#     '''resume'''
#     ckpt_manager = CheckpointManager(model,
#                                      optimizer,
#                                      ema_model=model_ema,
#                                      save_dir=args.exp_dir,
#                                      rank=args.rank,
#                                      additions={
#                                          'scaler': loss_scaler,
#                                          'dyrep': dyrep
#                                      })

#     if args.resume:
#         start_epoch = ckpt_manager.load(args.resume) + 1
#         if start_epoch > args.warmup_epochs:
#             scheduler.finished = True
#         scheduler.step(start_epoch * len(train_loader))
#         if args.dyrep:
#             model = DDP(model.module,
#                         device_ids=[args.local_rank],
#                         find_unused_parameters=True)
#         logger.info(
#             f'Resume ckpt {args.resume} done, '
#             f'start training from epoch {start_epoch}'
#         )
#     else:
#         start_epoch = 0

#     '''auxiliary tower'''
#     if args.auxiliary:
#         auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
#     else:
#         auxiliary_buffer = None

#     '''train & val'''
#     for epoch in range(start_epoch, args.epochs):
#         train_loader.loader.sampler.set_epoch(epoch)

#         if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
#             # update drop path rate
#             if hasattr(model.module, 'drop_path_rate'):
#                 model.module.drop_path_rate = \
#                     args.drop_path_rate * epoch / args.epochs

#         # train
#         metrics = train_epoch(args, epoch, model, model_ema, train_loader,
#                               optimizer, loss_fn, scheduler, auxiliary_buffer,
#                               dyrep, loss_scaler, matryoshka_trainer)

#         # validate
#         test_metrics = validate(args, epoch, model, val_loader, val_loss_fn, matryoshka_trainer=matryoshka_trainer)
#         if model_ema is not None:
#             test_metrics = validate(args,
#                                     epoch,
#                                     model_ema.module,
#                                     val_loader,
#                                     val_loss_fn,
#                                     log_suffix='(EMA)',
#                                     matryoshka_trainer=matryoshka_trainer)

#         # dyrep
#         if dyrep is not None:
#             if epoch < args.dyrep_max_adjust_epochs:
#                 if (epoch + 1) % args.dyrep_adjust_interval == 0:
#                     # adjust
#                     logger.info('DyRep: adjust model.')
#                     dyrep.adjust_model()
#                     logger.info(
#                         f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
#                     )
#                     # re-init DDP
#                     model = DDP(model.module,
#                                 device_ids=[args.local_rank],
#                                 find_unused_parameters=True)
#                     test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
#                 elif args.dyrep_recal_bn_every_epoch:
#                     logger.info('DyRep: recalibrate BN.')
#                     recal_bn(model.module, train_loader, 200)
#                     test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

#         metrics.update(test_metrics)
#         ckpts = ckpt_manager.update(epoch, metrics)
#         logger.info('\n'.join(['Checkpoints:'] + [
#             '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
#         ]))


# def train_epoch(args,
#                 epoch,
#                 model,
#                 model_ema,
#                 loader,
#                 optimizer,
#                 loss_fn,
#                 scheduler,
#                 auxiliary_buffer=None,
#                 dyrep=None,
#                 loss_scaler=None,
#                 matryoshka_trainer=None):
#     loss_m = AverageMeter(dist=True)
#     data_time_m = AverageMeter(dist=True)
#     batch_time_m = AverageMeter(dist=True)
#     start_time = time.time()

#     model.train()
#     for batch_idx, (input, target) in enumerate(loader):
#         data_time = time.time() - start_time
#         data_time_m.update(data_time)

#         # optimizer.zero_grad()
#         # use optimizer.zero_grad(set_to_none=True) for speedup
#         for p in model.parameters():
#             p.grad = None

#         with torch.cuda.amp.autocast(enabled=loss_scaler is not None):
#             is_matryoshka = args.model.startswith('matryoshka_')
            
#             if not args.kd:
#                 if is_matryoshka:
#                     # Get training dimensions for this epoch
#                     training_dims = matryoshka_trainer.get_training_dims(epoch)
#                     outputs = model(input, dims=training_dims)
#                     loss = loss_fn(outputs, target)
#                     # Use largest dimension for metrics
#                     largest_dim = max(outputs.keys())
#                     output = outputs[largest_dim]
#                 else:
#                     output = model(input)
#                     loss = loss_fn(output, target)
#             else:
#                 loss = loss_fn(input, target)
    
#             if auxiliary_buffer is not None:
#                 loss_aux = loss_fn(auxiliary_buffer.output, target)
#                 loss += loss_aux * auxiliary_buffer.loss_weight

#         if loss_scaler is None:
#             loss.backward()
#         else:
#             # amp
#             loss_scaler.scale(loss).backward()
#         if args.clip_grad_norm:
#             if loss_scaler is not None:
#                 loss_scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                            args.clip_grad_max_norm)

#         if dyrep is not None:
#             # record states of model in dyrep
#             dyrep.record_metrics()
            
#         if loss_scaler is None:
#             optimizer.step()
#         else:
#             loss_scaler.step(optimizer)
#             loss_scaler.update()

#         if model_ema is not None:
#             model_ema.update(model)

#         loss_m.update(loss.item(), n=input.size(0))
#         batch_time = time.time() - start_time
#         batch_time_m.update(batch_time)
#         if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
#             is_matryoshka = args.model.startswith('matryoshka_')
            
#             if _has_nvidia_smi:
#                 util = int(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
#                 mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
                
#                 if is_matryoshka and hasattr(loss_fn, 'loss_dict'):
#                     # Log individual dimension losses
#                     dim_losses = ' '.join([f'{k}: {v:.3f}' for k, v in loss_fn.loss_dict.items()])
#                     logger.info('Train: {} [{:>4d}/{}] '
#                                 'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
#                                 'LR: {lr:.3e} '
#                                 'Dims: [{dims}] '
#                                 'Mem: {memory:.0f} '
#                                 'Util: {util:d}% '
#                                 'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
#                                 'Data: {data_time.val:.2f}s'.format(
#                                     epoch,
#                                     batch_idx,
#                                     len(loader),
#                                     loss=loss_m,
#                                     lr=optimizer.param_groups[0]['lr'],
#                                     dims=dim_losses,
#                                     util=util,
#                                     memory=mem,
#                                     batch_time=batch_time_m,
#                                     data_time=data_time_m))
#                 else:
#                     logger.info('Train: {} [{:>4d}/{}] '
#                                 'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
#                                 'LR: {lr:.3e} '
#                                 'Mem: {memory:.0f} '
#                                 'Util: {util:d}% '
#                                 'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
#                                 'Data: {data_time.val:.2f}s'.format(
#                                     epoch,
#                                     batch_idx,
#                                     len(loader),
#                                     loss=loss_m,
#                                     lr=optimizer.param_groups[0]['lr'],
#                                     util=util,
#                                     memory=mem,
#                                     batch_time=batch_time_m,
#                                     data_time=data_time_m))
#             else:
#                 if is_matryoshka and hasattr(loss_fn, 'loss_dict'):
#                     # Log individual dimension losses
#                     dim_losses = ' '.join([f'{k}: {v:.3f}' for k, v in loss_fn.loss_dict.items()])
#                     logger.info('Train: {} [{:>4d}/{}] '
#                                 'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
#                                 'LR: {lr:.3e} '
#                                 'Dims: [{dims}] '
#                                 'Mem: {memory:.0f} '
#                                 'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
#                                 'Data: {data_time.val:.2f}s'.format(
#                                     epoch,
#                                     batch_idx,
#                                     len(loader),
#                                     loss=loss_m,
#                                     lr=optimizer.param_groups[0]['lr'],
#                                     dims=dim_losses,
#                                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
#                                     batch_time=batch_time_m,
#                                     data_time=data_time_m))
#                 else:
#                     logger.info('Train: {} [{:>4d}/{}] '
#                                 'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
#                                 'LR: {lr:.3e} '
#                                 'Mem: {memory:.0f} '
#                                 'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
#                                 'Data: {data_time.val:.2f}s'.format(
#                                     epoch,
#                                     batch_idx,
#                                     len(loader),
#                                     loss=loss_m,
#                                     lr=optimizer.param_groups[0]['lr'],
#                                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
#                                     batch_time=batch_time_m,
#                                     data_time=data_time_m))
#         scheduler.step(epoch * len(loader) + batch_idx + 1)
#         start_time = time.time()

#     return {'train_loss': loss_m.avg}


# def validate(args, epoch, model, loader, loss_fn, log_suffix='', matryoshka_trainer=None):
#     loss_m = AverageMeter(dist=True)
#     top1_m = AverageMeter(dist=True)
#     top5_m = AverageMeter(dist=True)
#     batch_time_m = AverageMeter(dist=True)
#     start_time = time.time()
    
#     is_matryoshka = args.model.startswith('matryoshka_')
#     all_accuracies = {}

#     model.eval()
#     for batch_idx, (input, target) in enumerate(loader):
#         with torch.no_grad():
#             if is_matryoshka:
#                 # Evaluate all dimensions during validation
#                 outputs = model(input)
#                 loss = loss_fn(outputs, target)
                
#                 # Compute accuracies for all dimensions
#                 if matryoshka_trainer:
#                     from lib.utils.matryoshka_utils import accuracy as matryoshka_accuracy
#                     batch_accs = matryoshka_trainer.compute_accuracy(outputs, target)
#                     for key, val in batch_accs.items():
#                         if key not in all_accuracies:
#                             all_accuracies[key] = AverageMeter(dist=True)
#                         all_accuracies[key].update(val, n=input.size(0))
                
#                 # Use largest dimension for main metrics
#                 largest_dim = max(outputs.keys())
#                 output = outputs[largest_dim]
#             else:
#                 output = model(input)
#                 loss = loss_fn(output, target)

#         top1, top5 = accuracy(output, target, topk=(1, 5))
#         loss_m.update(loss.item(), n=input.size(0))
#         top1_m.update(top1 * 100, n=input.size(0))
#         top5_m.update(top5 * 100, n=input.size(0))

#         batch_time = time.time() - start_time
#         batch_time_m.update(batch_time)
#         if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
#             logger.info('Test{}: {} [{:>4d}/{}] '
#                         'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
#                         'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
#                         'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
#                         'Time: {batch_time.val:.2f}s'.format(
#                             log_suffix,
#                             epoch,
#                             batch_idx,
#                             len(loader),
#                             loss=loss_m,
#                             top1=top1_m,
#                             top5=top5_m,
#                             batch_time=batch_time_m))
#         start_time = time.time()

#     results = {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}
    
#     # Add Matryoshka dimension results
#     if is_matryoshka and all_accuracies:
#         for key, meter in all_accuracies.items():
#             results[key] = meter.avg
    
#     return results


# if __name__ == '__main__':
#     main()
import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

try:
    # need `pip install nvidia-ml-py3` to measure gpu stats
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    _has_nvidia_smi = True
except ModuleNotFoundError:
    _has_nvidia_smi = False


torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    # save args
    logger.info(args)
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)

    '''build model'''
    # Check if this is a Matryoshka model
    is_matryoshka = args.model.startswith('matryoshka_')
    
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        base_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        base_loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        base_loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                              epsilon=args.smoothing).cuda()
    
    # Initialize Matryoshka trainer if needed
    matryoshka_trainer = None
    if is_matryoshka:
        from lib.utils.matryoshka_utils import MatryoshkaTrainer
        from lib.models.losses import MatryoshkaLoss
        
        # Use Matryoshka loss
        loss_fn = MatryoshkaLoss(
            matryoshka_dims=args.matryoshka_dims,
            loss_weights=args.matryoshka_loss_weights,
            base_loss_fn=base_loss_fn,
            label_smoothing=args.smoothing
        ).cuda()
        
        # Initialize Matryoshka trainer
        matryoshka_trainer = MatryoshkaTrainer(
            matryoshka_dims=args.matryoshka_dims,
            adaptive_training=args.matryoshka_adaptive,
            warmup_epochs=args.matryoshka_warmup
        )
    else:
        loss_fn = base_loss_fn
    
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    # logger.info(
    #     f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M')

    # Diverse Branch Blocks
    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    model.cuda()
    

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(model, teacher_model, args.model, args.teacher_model, loss_fn, 
                         args.kd, args.ori_loss_weight, args.kd_loss_weight, args.kd_loss_kwargs)

    # Check if this is a Matryoshka model for DDP configuration
    is_matryoshka = args.model.startswith('matryoshka_')
    
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=is_matryoshka)  # Enable for Matryoshka models
    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    model.register_comm_hook(None, comm_hooks.fp16_compress_hook)
    if args.kd != '':
        loss_fn.student = model
    logger.info(model)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''build optimizer'''
    optimizer = build_optimizer(args.opt,
                                model.module,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)

    '''build scheduler'''
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(args.sched,
                                optimizer,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn
        dyrep = DyRep(
            model.module,
            optimizer,
            recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
                                           args.dyrep_recal_bn_iters, m),
            filter_bias_and_bn=not args.opt_no_filter)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'dyrep': dyrep
                                     })

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
        scheduler.step(start_epoch * len(train_loader))
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )
    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None

    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs

        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader,
                              optimizer, loss_fn, scheduler, auxiliary_buffer,
                              dyrep, loss_scaler, matryoshka_trainer)

        # validate
        test_metrics = validate(args, epoch, model, val_loader, val_loss_fn, matryoshka_trainer=matryoshka_trainer)
        if model_ema is not None:
            test_metrics = validate(args,
                                    epoch,
                                    model_ema.module,
                                    val_loader,
                                    val_loss_fn,
                                    log_suffix='(EMA)',
                                    matryoshka_trainer=matryoshka_trainer)

        # dyrep
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    # adjust
                    logger.info('DyRep: adjust model.')
                    dyrep.adjust_model()
                    logger.info(
                        f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
                    )
                    # re-init DDP
                    model = DDP(model.module,
                                device_ids=[args.local_rank],
                                find_unused_parameters=True)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info('DyRep: recalibrate BN.')
                    recal_bn(model.module, train_loader, 200)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics)
        logger.info('\n'.join(['Checkpoints:'] + [
            '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
        ]))


def train_epoch(args,
                epoch,
                model,
                model_ema,
                loader,
                optimizer,
                loss_fn,
                scheduler,
                auxiliary_buffer=None,
                dyrep=None,
                loss_scaler=None,
                matryoshka_trainer=None):
    loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    for batch_idx, (input, target) in enumerate(loader):
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=True) for speedup
        for p in model.parameters():
            p.grad = None

        with torch.amp.autocast('cuda', enabled=loss_scaler is not None):
            is_matryoshka = args.model.startswith('matryoshka_')
            outputs_for_acc = None  # Store outputs for accuracy calculation
            
            if not args.kd:
                if is_matryoshka:
                    # Always compute all dimensions during training to use all parameters
                    outputs = model(input)  # Use all dimensions
                    outputs_for_acc = outputs  # Store for accuracy calculation
                    
                    # Get training dimensions for this epoch (for loss computation)
                    training_dims = matryoshka_trainer.get_training_dims(epoch)
                    # Filter outputs to only include training dimensions for loss
                    training_outputs = {dim: outputs[dim] for dim in training_dims if dim in outputs}
                    loss = loss_fn(training_outputs, target)
                    # Use largest dimension for metrics
                    largest_dim = max(outputs.keys())
                    output = outputs[largest_dim]
                else:
                    output = model(input)
                    loss = loss_fn(output, target)
            else:
                loss = loss_fn(input, target)
    
            if auxiliary_buffer is not None:
                loss_aux = loss_fn(auxiliary_buffer.output, target)
                loss += loss_aux * auxiliary_buffer.loss_weight

        if loss_scaler is None:
            loss.backward()
        else:
            # amp
            loss_scaler.scale(loss).backward()
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)

        if dyrep is not None:
            # record states of model in dyrep
            dyrep.record_metrics()
            
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        
        # Compute training accuracies for Matryoshka dimensions if needed for logging
        train_dim_accs = {}
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            is_matryoshka = args.model.startswith('matryoshka_')
            
            # Compute accuracy for each dimension during training (for logging only)
            if is_matryoshka and outputs_for_acc is not None:
                with torch.no_grad():
                    for dim, out in outputs_for_acc.items():
                        if out.size(0) == target.size(0):  # Check batch size matches
                            try:
                                # Safe accuracy calculation
                                _, pred = out.topk(1, 1, True, True)
                                pred = pred.squeeze()
                                if pred.dim() == 0:
                                    pred = pred.unsqueeze(0)
                                correct = pred.eq(target)
                                acc = correct.float().mean() * 100
                                train_dim_accs[dim] = acc.item()
                            except Exception as e:
                                # If accuracy calculation fails, skip this dimension
                                continue
            
            if _has_nvidia_smi:
                util = int(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
                mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
                
                if is_matryoshka and hasattr(loss_fn, 'loss_dict'):
                    # Log individual dimension losses and accuracies
                    dim_losses = ' '.join([f'{k}: {v:.3f}' for k, v in loss_fn.loss_dict.items()])
                    dim_accs = ' '.join([f'{dim}: {acc:.1f}%' for dim, acc in train_dim_accs.items()])
                    logger.info('Train: {} [{:>4d}/{}] '
                                'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                                'LR: {lr:.3e} '
                                'Dims Loss: [{dims_loss}] '
                                'Dims Acc: [{dims_acc}] '
                                'Mem: {memory:.0f} '
                                'Util: {util:d}% '
                                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                                'Data: {data_time.val:.2f}s'.format(
                                    epoch,
                                    batch_idx,
                                    len(loader),
                                    loss=loss_m,
                                    lr=optimizer.param_groups[0]['lr'],
                                    dims_loss=dim_losses,
                                    dims_acc=dim_accs,
                                    util=util,
                                    memory=mem,
                                    batch_time=batch_time_m,
                                    data_time=data_time_m))
                else:
                    logger.info('Train: {} [{:>4d}/{}] '
                                'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                                'LR: {lr:.3e} '
                                'Mem: {memory:.0f} '
                                'Util: {util:d}% '
                                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                                'Data: {data_time.val:.2f}s'.format(
                                    epoch,
                                    batch_idx,
                                    len(loader),
                                    loss=loss_m,
                                    lr=optimizer.param_groups[0]['lr'],
                                    util=util,
                                    memory=mem,
                                    batch_time=batch_time_m,
                                    data_time=data_time_m))
            else:
                if is_matryoshka and hasattr(loss_fn, 'loss_dict'):
                    # Log individual dimension losses and accuracies
                    dim_losses = ' '.join([f'{k}: {v:.3f}' for k, v in loss_fn.loss_dict.items()])
                    dim_accs = ' '.join([f'{dim}: {acc:.1f}%' for dim, acc in train_dim_accs.items()])
                    logger.info('Train: {} [{:>4d}/{}] '
                                'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                                'LR: {lr:.3e} '
                                'Dims Loss: [{dims_loss}] '
                                'Dims Acc: [{dims_acc}] '
                                'Mem: {memory:.0f} '
                                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                                'Data: {data_time.val:.2f}s'.format(
                                    epoch,
                                    batch_idx,
                                    len(loader),
                                    loss=loss_m,
                                    lr=optimizer.param_groups[0]['lr'],
                                    dims_loss=dim_losses,
                                    dims_acc=dim_accs,
                                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                                    batch_time=batch_time_m,
                                    data_time=data_time_m))
                else:
                    logger.info('Train: {} [{:>4d}/{}] '
                                'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                                'LR: {lr:.3e} '
                                'Mem: {memory:.0f} '
                                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                                'Data: {data_time.val:.2f}s'.format(
                                    epoch,
                                    batch_idx,
                                    len(loader),
                                    loss=loss_m,
                                    lr=optimizer.param_groups[0]['lr'],
                                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                                    batch_time=batch_time_m,
                                    data_time=data_time_m))
        scheduler.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()

    return {'train_loss': loss_m.avg}


def validate(args, epoch, model, loader, loss_fn, log_suffix='', matryoshka_trainer=None):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()
    
    is_matryoshka = args.model.startswith('matryoshka_')
    all_accuracies = {}

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            if is_matryoshka:
                # Evaluate all dimensions during validation
                outputs = model(input)
                loss = loss_fn(outputs, target)
                
                # Compute accuracies for all dimensions
                if matryoshka_trainer:
                    from lib.utils.matryoshka_utils import accuracy as matryoshka_accuracy
                    batch_accs = matryoshka_trainer.compute_accuracy(outputs, target)
                    for key, val in batch_accs.items():
                        if key not in all_accuracies:
                            all_accuracies[key] = AverageMeter(dist=True)
                        all_accuracies[key].update(val, n=input.size(0))
                
                # Use largest dimension for main metrics
                largest_dim = max(outputs.keys())
                output = outputs[largest_dim]
            else:
                output = model(input)
                loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    results = {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}
    
    # Add Matryoshka dimension results
    if is_matryoshka and all_accuracies:
        for key, meter in all_accuracies.items():
            results[key] = meter.avg
        
        # Log individual Matryoshka dimension accuracies
        dim_top1_results = []
        dim_top5_results = []
        for key, meter in all_accuracies.items():
            if 'top1_dim' in key:
                dim_name = key.replace('top1_dim_', '')
                dim_top1_results.append(f'{dim_name}: {meter.avg:.2f}%')
            elif 'top5_dim' in key:
                dim_name = key.replace('top5_dim_', '')
                dim_top5_results.append(f'{dim_name}: {meter.avg:.2f}%')
        
        if dim_top1_results:
            logger.info(f'Matryoshka Top-1{log_suffix}: {" | ".join(dim_top1_results)}')
        if dim_top5_results:
            logger.info(f'Matryoshka Top-5{log_suffix}: {" | ".join(dim_top5_results)}')
    
    return results


if __name__ == '__main__':
    main()
