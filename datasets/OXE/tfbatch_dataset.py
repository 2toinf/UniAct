from datasets.OXE.transforms import OXE_STANDARDIZATION_TRANSFORMS, chunk_act_obs
from datasets.OXE.configs import OXE_DATASET_CONFIGS
from datasets.OXE.mixture import OXE_NAMED_MIXTURES
from datasets.OXE.action_statics import OXE_ACTION_STATICS
import tensorflow_io as tfio # MUST import to enable file system support.
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
from torch.utils.data import IterableDataset,DataLoader
import torch
from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers import LlavaOnevisionProcessor
from functools import partial
import os
import numpy as np

os.environ['AWS_ACCESS_KEY_ID'] = 'P23QN96R08I1YOIVNNV3'           
os.environ['AWS_SECRET_ACCESS_KEY'] = '0HxxzVxFyDRGIRBUfoYVORK6tE537yVU5aZxhVK8'           
os.environ['S3_ENDPOINT'] = 'http://10.140.14.204:80'
os.environ['S3_USE_HTTPS'] = '0'
os.environ['S3_VERIFY_SSL'] = '0'

LLAVAOV_PREPROCESSOR = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf",local_files_only = True)
S3Path = "s3://real_data_raw/open_x_embodiment_origin/"
LOCAL_OXE = '/mnt/hwfile/OpenRobotLab/robot_data/open_x_embodiment_origin_re/'

def traj_standarize(traj, 
                    dataset_name, 
                    window_size = 1,
                    action_chunk_length = 3):
    traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)
    traj = chunk_act_obs(traj, window_size=window_size, future_action_window_size=action_chunk_length)
    return {
        'instruction': traj['language_instruction'],
        "image": traj['observation'][OXE_DATASET_CONFIGS[dataset_name]['image_obs_keys']['primary']],
        'action': traj['action']
    }

def frame_standarize_with_img_aug(frame, statics, method):
    # image augmentation
    augment_kwargs = dict(
                random_resized_crop=dict(scale=[0.7, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
    )
    frame['image'] = tf.io.decode_image(frame['image'][0], expand_animations=False, dtype=tf.uint8)
    frame['image'] = dl.transforms.resize_image(frame['image'], [384, 384])
    frame['image'] = dl.transforms.augment_image(frame['image'], **augment_kwargs)
    # action mask
    action_dim_mask = not tf.reduce_all(tf.equal(frame['action'], 0), axis=-1)
    hidden_dim_mask = not tf.equal(statics['99max'] - statics['1min'], 0.0)
    frame['action_mask'] = tf.logical_and(tf.expand_dims(action_dim_mask, axis=-1),
                                          hidden_dim_mask)
    
    # action normalization
    if method == 'min_max_99':
        frame['action'] = (tf.cast(frame['action'], tf.float32) - statics['1min']) / \
                        (statics['99max'] - statics['1min'] + 1e-6)
    else:
        raise NotImplementedError
    
    return frame
    

def dataset2path(dataset_name):
    if dataset_name == 'bridge_dataset' or dataset_name == 'droid': version = '1.0.0'
    elif dataset_name == 'dobbe' or dataset_name == 'fmb_dataset': version = '0.0.1'
    else: version = '0.1.0'
    base_path = LOCAL_OXE if dataset_name in os.listdir(LOCAL_OXE) else S3Path

    return base_path + f'{dataset_name}/{version}'

def OXEIterator(
            dataset_name,
            batch_size,
            action_chunk_length = 3,
            action_normalize_way = 'min_max_99',
            shuffle_buffer_size = 64):
    
    action_chunk_length -= 1
    action_chunk_length = action_chunk_length
    action_normalize_way = action_normalize_way
    action_statics = OXE_ACTION_STATICS[dataset_name]
    for key, value in action_statics.items():
        action_statics[key] = tf.constant(value, dtype=tf.float32)
    
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False, num_parallel_reads=1).traj_map(
        partial(traj_standarize, 
                dataset_name = dataset_name, 
                action_chunk_length = action_chunk_length),
        num_parallel_calls=1,
    ).flatten(num_parallel_calls=1).frame_map(
        partial(
            frame_standarize_with_img_aug,
            statics = action_statics,
            method = action_normalize_way
        ),
        num_parallel_calls=2
    ).repeat().batch(batch_size, num_parallel_calls=2).shuffle(shuffle_buffer_size).with_ram_budget(1).ignore_errors()

    for batch in dataset.as_numpy_iterator():
        text = [LLAVAOV_PREPROCESSOR.apply_chat_template([
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text":  item.decode("utf-8")},
                ]
            }], add_generation_prompt=True) for item in batch['instruction']]

        video = [np.expand_dims(meta, axis=0) for meta in batch['image']]
        inputs = LLAVAOV_PREPROCESSOR(videos=video, text=text, return_tensors="pt", padding=True)
        yield  {
            'inputs': inputs,
            'action': torch.from_numpy(batch['action']),
            'action_mask': torch.from_numpy(batch['action_mask'])
        }


def create_OXE_datasets_tf(
        batch_size, 
        action_chunk_length,
        use_recipe = 'oxe_magic_soup',   
        **kwargs):
    sample_weight_dict = {}
    dataloader_dict = {}
    for dataset_name, weight in OXE_NAMED_MIXTURES[use_recipe]:
        sample_weight_dict[dataset_name] = weight
        dataloader_dict[dataset_name] = OXEIterator(dataset_name,
                                                   batch_size=batch_size,
                                                   action_chunk_length=action_chunk_length)
    return sample_weight_dict, dataloader_dict
