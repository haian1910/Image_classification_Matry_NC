import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def compute_teacher_targets_inline(teacher_model, train_loader, val_loader, args, logger):
    """
    Compute teacher targets (class means and Gram matrix) inline during training setup.
    For now, we'll create reasonable dummy targets to get NC2 working.
    """
    try:
        logger.info("Creating dummy teacher targets for NC2...")
        
        # Get teacher feature dimension by doing a forward pass
        teacher_model.eval()
        with torch.no_grad():
            # Get a sample batch
            sample_batch = next(iter(val_loader))
            sample_images = sample_batch[0][:1].cuda()  # Just one image
            
            # Hook to capture features
            captured_features = []
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() > 2:
                    output = torch.flatten(output, 1)
                captured_features.append(output.shape[1])  # Store feature dimension
            
            # Register hook
            hook_registered = False
            hook = None
            for name, module in teacher_model.named_modules():
                if 'avgpool' in name or isinstance(module, nn.AdaptiveAvgPool2d):
                    hook = module.register_forward_hook(capture_hook)
                    hook_registered = True
                    logger.info(f"Registered hook at: {name}")
                    break
            
            if not hook_registered:
                # Fallback to last layer before classifier
                for name, module in teacher_model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)) and 'fc' not in name.lower():
                        hook = module.register_forward_hook(capture_hook)
                        hook_registered = True
                        logger.info(f"Registered hook at: {name}")
                        break
            
            if hook_registered:
                _ = teacher_model(sample_images)
                feature_dim = captured_features[0] if captured_features else 64
                hook.remove()
            else:
                logger.warning("Could not register hook, using default feature dimension")
                feature_dim = 64
        
        num_classes = getattr(args, 'num_classes', 100)
        
        logger.info(f"Using feature dimension: {feature_dim}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Create dummy class means (random but consistent)
        torch.manual_seed(42)  # For reproducibility
        class_means = torch.randn(num_classes, feature_dim)
        # Normalize each class mean
        class_means = F.normalize(class_means, p=2, dim=1)
        
        # Compute Gram matrix
        gram_matrix = torch.mm(class_means, class_means.t())
        gram_matrix = gram_matrix / (torch.norm(gram_matrix, p='fro') + 1e-8)
        
        logger.info(f"Created dummy class means: {class_means.shape}")
        logger.info(f"Created dummy Gram matrix: {gram_matrix.shape}")
        
        return {
            'class_means': class_means,
            'gram_matrix': gram_matrix
        }
        
    except Exception as e:
        logger.error(f"Error creating dummy teacher targets: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def compute_teacher_targets_inline(teacher_model, train_loader, val_loader, args, logger):
    """
    Compute teacher targets (class means and Gram matrix) inline during training setup.
    This is a simplified version that computes targets from validation data.
    """
    try:
        logger.info("Computing teacher class means and Gram matrix...")
        teacher_model.eval()
        
        # Use validation loader for faster computation
        data_loader = val_loader
        
        # Storage for features and labels
        all_features = []
        all_labels = []
        
        # Hook to extract features from the teacher
        features_dict = {}
        
        def hook_fn(module, input, output):
            # Store the features
            feat = output[0] if isinstance(output, tuple) else output
            if feat.dim() > 2:
                feat = torch.flatten(feat, 1)  # Flatten spatial dimensions
            features_dict['features'] = feat.detach()
        
        # Register hook on teacher avgpool/final feature layer
        hook_registered = False
        for name, module in teacher_model.named_modules():
            if 'avgpool' in name or 'pool' in name or 'fc' in name:
                if not hook_registered:  # Only register on the first pooling layer found
                    handle = module.register_forward_hook(hook_fn)
                    hook_registered = True
                    logger.info(f"Registered hook on teacher module: {name}")
                    break
        
        if not hook_registered:
            logger.warning("Could not find suitable layer to hook in teacher model")
            return None
        
        device = next(teacher_model.parameters()).device
        num_samples = 0
        max_samples = min(1000, len(data_loader) * args.batch_size)  # Limit samples for speed
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                if num_samples >= max_samples:
                    break
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                _ = teacher_model(inputs)
                
                if 'features' in features_dict:
                    all_features.append(features_dict['features'].cpu())
                    all_labels.append(targets.cpu())
                    num_samples += inputs.size(0)
                    
                if i % 50 == 0:
                    logger.info(f"Processed {num_samples}/{max_samples} samples")
        
        # Remove hook
        handle.remove()
        
        if not all_features:
            logger.error("No features extracted from teacher model")
            return None
        
        # Concatenate all features and labels
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        logger.info(f"Extracted features: {features.shape}, labels: {labels.shape}")
        
        # Compute class means
        num_classes = args.num_classes
        feature_dim = features.size(1)
        
        class_means = torch.zeros(num_classes, feature_dim)
        class_counts = torch.zeros(num_classes)
        
        for class_id in range(num_classes):
            mask = (labels == class_id)
            if mask.sum() > 0:
                class_means[class_id] = features[mask].mean(dim=0)
                class_counts[class_id] = mask.sum()
        
        # Only keep classes that have samples
        valid_classes = class_counts > 0
        if valid_classes.sum() < num_classes:
            logger.warning(f"Only {valid_classes.sum()}/{num_classes} classes found in data")
        
        # Compute Gram matrix
        gram_matrix = torch.mm(class_means, class_means.t())
        # Normalize
        gram_norm = torch.norm(gram_matrix, p='fro')
        if gram_norm > 0:
            gram_matrix = gram_matrix / gram_norm
        
        logger.info(f"Computed class means: {class_means.shape}")
        logger.info(f"Computed Gram matrix: {gram_matrix.shape}")
        
        return {
            'class_means': class_means,
            'gram_matrix': gram_matrix,
            'num_classes': num_classes,
            'feature_dim': feature_dim,
            'class_counts': class_counts
        }
        
    except Exception as e:
        logger.error(f"Failed to compute teacher targets: {e}")
        return None


torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def compute_teacher_targets_inline(teacher_model, train_loader, val_loader, args, logger):
    """
    Compute teacher targets (class means and Gram matrix) for NC2 inline during training.
    """
    try:
        from collections import defaultdict
        
        logger.info("Computing teacher targets for NC2...")
        teacher_model.eval()
        
        # Extract features using hooks
        features_list = []
        labels_list = []
        
        def hook_fn(module, input, output):
            # Flatten spatial dimensions if needed
            if output.dim() > 2:
                output = torch.flatten(output, 1)
            features_list.append(output.detach().cpu())
        
        # Register hook on the appropriate layer (avgpool for most models)
        hook_target = None
        for name, module in teacher_model.named_modules():
            if 'avgpool' in name or (name == 'avgpool' and hasattr(module, 'forward')):
                hook_target = module
                break
        
        if hook_target is None:
            # Fallback: try to find the last pooling or linear layer before classifier
            for name, module in teacher_model.named_modules():
                if isinstance(module, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d, torch.nn.Linear)):
                    hook_target = module
        
        if hook_target is None:
            logger.error("Could not find suitable layer for feature extraction")
            return None
        
        hook = hook_target.register_forward_hook(hook_fn)
        
        # Extract features from validation set (smaller, faster)
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_loader):
                if batch_idx >= 50:  # Limit to 50 batches for speed
                    break
                    
                input = input.cuda()
                target = target.cuda()
                
                # Forward pass to trigger hook
                _ = teacher_model(input)
                
                if features_list:
                    labels_list.append(target.detach().cpu())
        
        hook.remove()
        
        if not features_list or not labels_list:
            logger.error("No features extracted from teacher model")
            return None
        
        # Concatenate all features and labels
        all_features = torch.cat(features_list, dim=0)  # [N, feature_dim]
        all_labels = torch.cat(labels_list, dim=0)      # [N]
        
        logger.info(f"Extracted {all_features.shape[0]} samples with {all_features.shape[1]} features")
        logger.info(f"Labels shape: {all_labels.shape}, unique labels: {len(all_labels.unique())}")
        logger.info(f"Label range: {all_labels.min().item()} to {all_labels.max().item()}")
        
        # Compute class means
        num_classes = args.num_classes
        class_means = torch.zeros(num_classes, all_features.shape[1])
        
        # Keep track of which classes we actually have samples for
        present_classes = all_labels.unique().tolist()
        logger.info(f"Classes present in data: {len(present_classes)} out of {num_classes}")
        
        for class_idx in range(num_classes):
            mask = (all_labels == class_idx)
            if mask.sum() > 0:
                class_means[class_idx] = all_features[mask].mean(dim=0)
            else:
                # If no samples for this class, use a small random vector
                class_means[class_idx] = torch.randn(all_features.shape[1]) * 0.01
        
        # Compute Gram matrix
        gram_matrix = torch.mm(class_means, class_means.t())
        
        # Normalize Gram matrix
        gram_norm = torch.norm(gram_matrix, p='fro')
        if gram_norm > 0:
            gram_matrix = gram_matrix / gram_norm
        
        teacher_targets = {
            'class_means': class_means,
            'gram_matrix': gram_matrix,
            'num_classes': num_classes,
            'feature_dim': all_features.shape[1]
        }
        
        logger.info(f"Teacher targets computed: {num_classes} classes, {all_features.shape[1]} feature dimensions")
        return teacher_targets
        
    except Exception as e:
        logger.error(f"Error computing teacher targets: {e}")
        return None


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

    # Check if this is a Matryoshka model for DDP configuration
    is_matryoshka = args.model.startswith('matryoshka_')
    
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=is_matryoshka)  # Enable for Matryoshka models
    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    model.register_comm_hook(None, comm_hooks.fp16_compress_hook)
    
    # knowledge distillation - create KDLoss after DDP wrapping
    if args.kd != '':
        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        
        # Prepare kd_loss_kwargs with defaults for NC1 and NC2
        if args.kd_loss_kwargs is None:
            if args.kd == 'nc1':
                # Default NC1 parameters
                kd_loss_kwargs = {
                    'num_classes': args.num_classes,
                    'temperature': 4.0,
                    'alpha': 0.7
                }
            elif args.kd == 'nc2':
                # Default NC2 parameters
                kd_loss_kwargs = {
                    'num_classes': args.num_classes,
                    'nc2_lambda': 1.0,      # Weight for NC2 Gram matrix loss
                    'nc2_alpha': 0.5,       # Balance between batch and EMA losses
                    'ema_momentum': 0.9,    # EMA momentum for class means
                    'epsilon': 1e-8         # Numerical stability
                }
            elif args.kd == 'nc':
                # Default combined NC (NC1 + NC2) parameters
                kd_loss_kwargs = {
                    'num_classes': args.num_classes,
                    'temperature': 4.0,     # For NC1
                    'alpha': 0.7,           # For NC1
                    'nc2_lambda': 1.0,      # Weight for NC2 Gram matrix loss
                    'nc2_alpha': 0.5,       # Balance between batch and EMA losses
                    'ema_momentum': 0.9,    # EMA momentum for class means
                    'epsilon': 1e-8,        # Numerical stability
                    'nc1_weight': 1.0,      # Weight for NC1 loss in combination
                    'nc2_weight': 1.0  ,     # Weight for NC2 loss in combination
                    'ortho_lambda': 1.0,    # Weight for orthogonality regularization
                    'ema_momentum': 0.95,   # EMA momentum for class means
                    'nc2_alpha': 0.5        # Interpolation between batch and EMA losses
                }
            else:
                kd_loss_kwargs = {}
        else:
            kd_loss_kwargs = args.kd_loss_kwargs
            
        loss_fn = KDLoss(model, teacher_model, args.model, args.teacher_model, loss_fn, 
                         args.kd, args.ori_loss_weight, args.kd_loss_weight, kd_loss_kwargs)
        loss_fn.student = model
        
        # Load pre-computed teacher targets for NC2 and combined NC
        if args.kd in ['nc2', 'nc']:
            teacher_targets_path = getattr(args, 'nc2_teacher_targets', None)
            
            # If not explicitly provided, try to construct the path from teacher model and dataset
            if teacher_targets_path is None:
                teacher_targets_path = f"./teacher_targets/{args.teacher_model}_{args.dataset}_targets.pth"
            
            if teacher_targets_path and os.path.exists(teacher_targets_path):
                logger.info(f"Loading NC2 teacher targets from {teacher_targets_path}")
                teacher_targets = torch.load(teacher_targets_path, map_location='cpu')
                
                # Move to GPU if available
                device = next(model.parameters()).device
                teacher_class_means = teacher_targets['class_means'].to(device)
                teacher_gram = teacher_targets['gram_matrix'].to(device)
                
                # Set targets in NC2 loss
                loss_fn.set_teacher_targets(teacher_class_means, teacher_gram)
                logger.info("NC2 teacher targets loaded successfully")
                logger.info(f"Loaded {teacher_class_means.shape[0]} classes with {teacher_class_means.shape[1]} features")
            else:
                logger.warning(f"NC2 teacher targets not found at {teacher_targets_path}.")
                logger.info("Computing teacher targets automatically...")
                
                # Automatically compute teacher targets
                teacher_targets = compute_teacher_targets_inline(
                    teacher_model, train_loader, val_loader, args, logger
                )
                
                if teacher_targets is not None:
                    # Save for future use
                    os.makedirs(os.path.dirname(teacher_targets_path), exist_ok=True)
                    torch.save(teacher_targets, teacher_targets_path)
                    logger.info(f"Teacher targets computed and saved to {teacher_targets_path}")
                    
                    # Set targets in NC2 loss
                    device = next(model.parameters()).device
                    teacher_class_means = teacher_targets['class_means'].to(device)
                    teacher_gram = teacher_targets['gram_matrix'].to(device)
                    
                    loss_fn.set_teacher_targets(teacher_class_means, teacher_gram)
                    logger.info("NC2 teacher targets loaded successfully")
                    logger.info(f"Loaded {teacher_class_means.shape[0]} classes with {teacher_class_means.shape[1]} features")
                else:
                    logger.warning("Failed to compute teacher targets. NC2 will use zero loss.")
    
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
    
    # Check if the actual model being validated is a Matryoshka model
    # Handle DDP-wrapped models by checking the underlying module
    unwrapped_model = model.module if hasattr(model, 'module') else model
    is_matryoshka = (hasattr(unwrapped_model, 'matryoshka_dims') or 
                     hasattr(unwrapped_model, 'matryoshka_head') or
                     'MatryoshkaModel' in str(type(unwrapped_model)) or
                     'matryoshka' in str(type(unwrapped_model)).lower())
    all_accuracies = {}

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            if is_matryoshka:
                # Evaluate all dimensions during validation
                outputs = model(input)
                loss = loss_fn(outputs, target)
                print("="*20)
                print("og loss: ",loss)
                
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
                outputs = model(input)
                
                # Handle both dict and tensor outputs
                if isinstance(outputs, dict):
                    # If model returns dict, use largest dimension
                    largest_dim = max(outputs.keys())
                    output = outputs[largest_dim]
                else:
                    output = outputs
                    
                loss = loss_fn(outputs, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        
        # Handle loss - it might already be a float or a tensor
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = loss
            
        loss_m.update(loss_value, n=input.size(0))
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
