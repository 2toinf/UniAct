import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tensorflow as tf
tf.config.set_visible_devices(devices=[], device_type='GPU')

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models.UniAct
from utils import MultiDataIterMetricLogger
from datasets.OXE.dataset import create_OXE_datasets
from datasets.AIRData.dataset import create_AIR_datasets
from pathlib import Path
from tensorboardX import SummaryWriter
import datetime
import argparse
import utils
import subprocess
import logging
import random
import deepspeed
import time
from timm.models import create_model



def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    
    # Base Settings
    parser.add_argument('--model', default="UniAct_05B_CodeBook_256", type=str)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--grad_accumulation_steps', default=1, type=int)
    parser.add_argument('--iters', default=1e6, type=int)
    parser.add_argument('--initial_t', default=2.0, type=float)
    parser.add_argument('--final_t', default=0.1, type=float)
    
    # Optimizer parameters
    parser.add_argument('--precision', default="bf16")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    
    # Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=10000, type=int,
                        help='(default: 10000iter)')
    parser.add_argument('--output_dir', default='runnings/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_iters', default=0, type=int, metavar='N',
                        help='start epoch')

    # DataLoader parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29529, type=int, help='port')

    return parser

def main(args):

    output_dir = Path(args.output_dir)
    tb_logger = None
    if args.rank == 0:
        tensorboard_path = os.path.join(output_dir, 'events')
        tb_logger = SummaryWriter(tensorboard_path)
    utils.init_log(__name__, log_file=os.path.join(output_dir, 'full_log.txt'), rank=args.rank)
    logger = logging.getLogger(__name__)
    print = logger.info
    print(args)
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu))
    seed = args.seed + args.rank


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    cudnn.benchmark = True

    print('========== init model and dataset ==========')
    model = create_model(args.model,
                max_steps = args.iters, 
                start_iters = args.start_iters,
                initial_t = args.initial_t,
                final_t = args.final_t).cuda()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        new_state_dict = {}
        model_state_dict = model.state_dict()
        for key, value in ckpt['module'].items():
            if key not in model_state_dict: continue
            if model_state_dict[key].shape != value.shape: continue
            new_state_dict[key] = value

        print(model.load_state_dict(new_state_dict, strict=False))
        print("==========resume training from {}==========".format(args.resume))
    
    ### init OXE dataset
    oxe_sample_weight_dict, oxe_dataloader_dict = create_OXE_datasets(
                batch_size=args.batch_size, 
                action_chunk_length=4)
    
    print("==========OXE dataset initialized==========")
    
    ### init AIR dataset
    air_sample_weight_dict, air_dataloader_dict = create_AIR_datasets(
                batch_size=args.batch_size, 
                action_chunk_length=4)
    
    print("==========AIR dataset initialized==========")
    
    
    sample_weight_dict = {**oxe_sample_weight_dict, **air_sample_weight_dict}
    dataloader_dict = {**oxe_dataloader_dict, **air_dataloader_dict}
    
    
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": (args.beta1, args.beta2),
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 0
        },
    }
    
    print('========== init deepspeed ==========')
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    

    print(f"========== iters start from {args.start_iters} ==========")
    start_time = time.time()
    global_idx = args.start_iters
    metric_logger = MultiDataIterMetricLogger(delimiter="  ")
    model_engine.train()
    for batch, domain_name in metric_logger.log_every(args.iters, sample_weight_dict, dataloader_dict, 10):
        inputs = {'inputs': batch['inputs'].to('cuda',torch.bfloat16), 
                'action': batch['action'],
                'action_mask': batch['action_mask']}
                
        loss, outputs = model_engine(domain_name=domain_name, 
                            log_file = os.path.join(output_dir, 'code.log'), 
                            **inputs)
        model_engine.backward(loss)
        model_engine.step()

        metric_logger.update(**outputs)
        # try: metric_logger.update(lr=model_engine.lr_scheduler.get_last_lr()[0])
        # except: pass
        if tb_logger is not None and deepspeed.dist.get_rank() == 0 and global_idx % 50 == 0:
            for k, meter in metric_logger.meters.items():
                tb_logger.add_scalar('train/{}_val'.format(k), meter.value, global_idx)
        if global_idx % args.save_interval == 0 and global_idx != 0:
            deepspeed.dist.barrier()
            model_engine.save_checkpoint(os.path.join(output_dir, f"ckpt"))
        global_idx += 1

    deepspeed.dist.barrier()
    model_engine.save_checkpoint(os.path.join(output_dir, f"ckpt"))

    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def slurm_env_init(args):
    args.rank = int(os.environ['SLURM_PROCID'])
    args.gpu = args.rank % torch.cuda.device_count()
    args.world_size = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_RANK'] = str(args.rank % num_gpus)
    os.environ['RANK'] = str(args.rank)
    torch.cuda.set_device(args.gpu)
    return args

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(slurm_env_init(args))