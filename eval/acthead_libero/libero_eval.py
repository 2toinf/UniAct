import os
from PIL import Image

import torch
import torchvision.transforms as transforms
import numpy as np
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from data.AIRData.config import AIRDATA_CONFIG

from tqdm import tqdm
import json

import multiprocessing
from functools import partial
import imageio

EPS = 1e-5
LIBERO_DATASETS = {'libero_goal': ["libero_goal"],
                   "libero_object": ["libero_object"],
                   "libero_spatial": ["libero_spatial"],
                   "libero_10": ["libero_10"],
                   "libero_90": ["libero_90"],
                   "libero30": ["libero_goal", "libero_object", "libero_spatial"],
                   "libero130": ["libero_90", "libero_goal", "libero_object", "libero_spatial", "libero_10"]}


benchmark_dict = benchmark.get_benchmark_dict()

def has_normalize(transform):
       if isinstance(transform, transforms.Compose):
              for t in transform.transforms:
                     if isinstance(t, transforms.Normalize):
                            return True
       return False


# transformation
from torchvision import transforms as T
import timm

a_min, a_max = np.array(AIRDATA_CONFIG['libero-2-rgb']['action_statics']['0min']), np.array(AIRDATA_CONFIG['libero-2-rgb']['action_statics']['100max'])
s_mean, s_std = np.array(AIRDATA_CONFIG['libero-2-rgb']['proprio_statics']['mean']), np.array(AIRDATA_CONFIG['libero-2-rgb']['proprio_statics']['std'])

class LIBEROEval():
       def __init__(
              self,
              task_suite_name: str, # can choose libero_spatial, libero_goal, libero_object, libero_10, libero30, libero_90, libero130.
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'],
              eval_horizon: int=600,
              num_episodes: int=1,  # 1 for single-process evaluation
              eval_freq: int=10,
              seed: int=42,
              llava_ov_process_fn = None,
              r18_process_fn = None,
       ):
              self.task_suite_name = task_suite_name
              
              assert self.task_suite_name in LIBERO_DATASETS
              self.task_list = LIBERO_DATASETS[self.task_suite_name]
              self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
              self.obs_key = obs_key
              
              self.llava_ov_process_fn = llava_ov_process_fn
              self.r18_process_fn = r18_process_fn
              
              self.eval_horizon = eval_horizon
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.seed = seed
              
              self.all_time_actions = np.zeros([1000, 1000+4, 7])
                     
       def _make_dir(self, save_path):
              task_suite_name = self.task_suite_name
              path = os.path.join(save_path, task_suite_name)
              if not os.path.exists(path):
                     os.makedirs(path)
              self.base_dir = path
              self.base_dir_video = os.path.join(self.base_dir, 'rollout_video')
              
              if not os.path.exists(self.base_dir_video):
                     os.makedirs(self.base_dir_video)
       
       def _init_env(self, task_suite, task_id: int=0):
              # get task information and env args
              task = task_suite.get_task(task_id)
              task_name = task.name
              task_description = task.language
              task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
              print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

              # step over the environment
              env_args = {
                     "bddl_file_name": task_bddl_file,
                     "camera_heights": 128,
                     "camera_widths": 128
              }
              
              # init thesubprocess vector environment
              env_num = self.num_episodes
              env = SubprocVectorEnv(
                     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
              )
              
              # environment reset 
              env.seed(self.seed + 100)
              env.reset()
              init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
              init_state_id = np.random.randint(0, init_states.shape[0], (1, ))
              obs = env.set_init_state(init_states[init_state_id])
              
              
              # return the environment
              env_dict = {}
              env_dict['env'] = env
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def _log_results(self, metrics: dict):
              print(metrics)
              save_name = os.path.join(self.base_dir, 'results.json')
              with open(save_name, 'a+') as f:
                     line = json.dumps(metrics)
                     f.write(line+'\n')
       
       def raw_obs_to_stacked_obs(self, obs, lang):
              env_num = len(obs)
              
              data = {
                     "obs": {},
                     "lang": lang,
              }
              
              for key in self.obs_key:
                     data["obs"][key] = []
                     
              for i in range(env_num):
                     for key in self.obs_key:
                            data['obs'][key].append(obs[i][key])
              
              for key in data['obs']:
                     data['obs'][key] = np.stack(data['obs'][key])
              
              return data     

       def _rollout(self, task_suite, policy, task_id: int=0, img_goal=False, use_ac=True, code_type="gumbel_hard", k=0., temperature=1.0):
              """
              rollout one episode and return the episode return
              """
              env = self._init_env(task_suite, task_id)
              lang = env['language_instruction']
              obs = env['obs']
              
              images = []
              
              for i in range(10):
                     action = np.array([0., 0., 0., 0., 0., 0., -1.0])[None, :]
                     obs, reward, done, info = env['env'].step(action)

              for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                     ## image
                     data = self.raw_obs_to_stacked_obs(obs, lang)
                     obs, lang = data['obs'], data['lang']
                     
                     # proprio
                     gripper_qpos = obs['robot0_gripper_qpos']
                     eef_pos = obs['robot0_eef_pos']
                     eef_quat = obs['robot0_eef_quat']
                     proprio = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1).reshape(1, -1)
                     proprio = torch.from_numpy((proprio - s_mean) / s_std)
                     
                     image = Image.fromarray(np.flip(np.flip(obs["agentview_image"], 1), 2).squeeze(0))
                     image_wrist = Image.fromarray(np.flip(np.flip(obs['robot0_eye_in_hand_image'], 1), 2).squeeze(0))
                     
                     image_np = np.asarray(image)[None, ...]
                     image_tensor = torch.stack([self.r18_process_fn(image), self.r18_process_fn(image_wrist)])
              
                     text = [self.llava_ov_process_fn.apply_chat_template([
                        {
                            "role": "user",
                            "content": [
                                {"type": "video"},
                                {"type": "text", "text":  lang},
                            ]
                        }], add_generation_prompt=True)]
                     video = [image_np]
                     inputs = self.llava_ov_process_fn(videos=video, text=text, return_tensors="pt", padding=True)
                     inputs = {'inputs': inputs.to("cuda:0").to(torch.bfloat16),
                               'images': image_tensor.unsqueeze(0).to('cuda:0', torch.bfloat16),
                               'proprios': proprio.to('cuda:0', torch.bfloat16),  
                               }

                     # reshape
                     action_predict = policy.infer(domain_name='libero-2-rgb', **inputs)    
                     action_predict = action_predict.reshape(-1, 7)[0].detach().to(torch.float32).cpu().numpy()
                     action_predict = (action_predict + 1) * (a_max - a_min) / 2 + a_min
                     action_predict[-1] = 1 if action_predict[-1] < 0.5 else -1
                     action = action_predict
                     
                     # print(action)
                     action = action[None, :]
                     
                     ### record the video
                     B, H, W, C = obs["agentview_image"].shape
                     images.append(np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C))
                     
                     # step
                     obs, reward, done, info = env['env'].step(action)
                     if done.all():
                            break
              
              save_path = f'{self.base_dir_video}/{lang}.mp4'
              self._save_video(save_path, images, done, fps=30)              
              
              num_success = 0
              for k in range(self.num_episodes):
                     num_success += int(done[k])
              avg_succ_rate = num_success / self.num_episodes
             
              metrics = {f'sim/{self.task_suite_name}/{lang}': avg_succ_rate}
              self._log_results(metrics)
              
              env['env'].close()
              return avg_succ_rate
       
       def _save_video(self, save_path: str, images: list, done: list, fps=30): 
              imageio.mimsave(save_path, images, fps=fps)
              
       
       def eval_episodes(self, policy, save_path: str, use_ac=True, num_episodes=5):
              """
              rollout several episodes and log the mean episode return
              """
              self._make_dir(save_path)
              
              rews = []
              policy.train()
              for task_suite in self.task_suite_list:
                     for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating..."):
                        r_cum = 0. 
                        for _ in range(num_episodes):
                            r_cum += self._rollout(task_suite, policy, task_id, use_ac)
                        r_cum /= num_episodes
                        rews.append(r_cum)
              eval_rewards = sum(rews) / len(rews)
              metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
              self._log_results(metrics)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()