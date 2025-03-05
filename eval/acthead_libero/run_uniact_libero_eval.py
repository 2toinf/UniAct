import sys
import os
sys.path.append(os.getcwd())

from timm.models import create_model
import torch
import argparse

from models.UniAct_V1 import UniAct
from data.utils import LLAVAOV_PREPROCESSOR, R18_PREPROCESSOR

def main():
    parser = argparse.ArgumentParser(description='single-process evaluation on Libero bench')
    parser.add_argument('--base_path', default='YOUR_BASEMODEL_PATH', type=str, help='vlm ckpt path')
    parser.add_argument('--head_path', default='YOUR_ACTION_HEAD_PATH', type=str, help='create model name')
    parser.add_argument('--num_episodes', default=10, type=int, help='evaluation episodes')
    parser.add_argument('--task_suites', default=['libero_spatial', 'libero_goal', 'libero_object', 'libero_10', 'libero_90'], nargs='+', help='base save path')

    args = parser.parse_args()
    kwargs = vars(args)
    
    # load model
    uniact_model = create_model("UniAct_05B_CodeBook_256_V1")
    base_path = kwargs['base_path']
    head_path = kwargs['head_path']
    print(uniact_model.load_state_dict(torch.load(base_path), strict=False))
    print(uniact_model.load_state_dict(torch.load(head_path), strict=False))

    # evaluator
    from libero_eval import LIBEROEval
    for task_suite in kwargs['task_suites']:
        if task_suite == 'libero_10':
            eval_horizon = 700
        else:
            eval_horizon = 300
        evaluator = LIBEROEval(task_suite_name=task_suite, 
                                eval_horizon=eval_horizon,
                                llava_ov_process_fn = LLAVAOV_PREPROCESSOR,
                                r18_process_fn = R18_PREPROCESSOR)

        save_path = os.path.join(os.getcwd(), 'results', 'libero')
        evaluator.eval_episodes(uniact_model.eval().to('cuda:0', torch.bfloat16),
                                save_path=save_path, 
                                num_episodes=kwargs['num_episodes'])  # not use this when use_ac=False
                    
if __name__ == '__main__':
    main()