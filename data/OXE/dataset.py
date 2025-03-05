from data.OXE.transforms import OXE_STANDARDIZATION_TRANSFORMS, chunk_act_obs
from data.OXE.configs import OXE_DATASET_CONFIGS
from data.OXE.mixture import OXE_NAMED_MIXTURES
from data.OXE.action_statics import OXE_ACTION_STATICS
import tensorflow_io as tfio # MUST import to enable file system support.
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
from torch.utils.data import IterableDataset,DataLoader
import torch
from transformers.feature_extraction_sequence_utils import BatchFeature
from functools import partial
import os
import numpy as np
from PIL import Image
from data.utils import LLAVAOV_PREPROCESSOR, R18_PREPROCESSOR
from transformers.feature_extraction_utils import BatchFeature

# NOTE: need to be filled
S3Path = ''
LOCAL_OXE = ''

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
                random_resized_crop=dict(scale=[0.7, 1.0], ratio=[1.0, 1.0]),
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
    elif method == 'mean_std':
            (tf.cast(frame['action'], tf.float32) - statics['mean']) / statics['std']
    else:
        raise NotImplementedError
    
    return frame
    

def dataset2path(dataset_name):
    if dataset_name == 'bridge_dataset' or dataset_name == 'droid': version = '1.0.0'
    elif dataset_name == 'dobbe' or dataset_name == 'fmb_dataset': version = '0.0.1'
    else: version = '0.1.0'
    base_path = LOCAL_OXE if dataset_name in os.listdir(LOCAL_OXE) else S3Path

    return base_path + f'{dataset_name}/{version}'

class OXEDataset(IterableDataset):
    def __init__(
                self,
                dataset_name,
                action_chunk_length = 3,
                action_normalize_way = 'min_max_99',
                shuffle_buffer_size = 64):
        super().__init__()
        action_chunk_length -= 1
        action_chunk_length = action_chunk_length
        action_normalize_way = action_normalize_way
        action_statics = OXE_ACTION_STATICS[dataset_name]
        for key, value in action_statics.items():
            action_statics[key] = tf.constant(value, dtype=tf.float32)
        
        builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
        self.dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False, num_parallel_reads=1).traj_map(
            partial(traj_standarize, 
                    dataset_name = dataset_name, 
                    action_chunk_length = action_chunk_length),
            num_parallel_calls=1,
        ).flatten(num_parallel_calls=1).repeat().shuffle(shuffle_buffer_size).frame_map(
            partial(
                frame_standarize_with_img_aug,
                statics = action_statics,
                method = action_normalize_way
            ),
            num_parallel_calls=1
        ).with_ram_budget(1).ignore_errors()

    def __iter__(self):
        for sample in self.dataset.as_numpy_iterator():
            yield {
                'instruction': sample['instruction'].decode("utf-8"),
                'image': sample['image'],
                'action': torch.from_numpy(sample['action']),
                'action_mask': torch.from_numpy(sample['action_mask'])
            }

def collate_fn(batch):
    text = [LLAVAOV_PREPROCESSOR.apply_chat_template([
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text":  item['instruction']},
                ]
            }], add_generation_prompt=True) for item in batch]

    video = [np.expand_dims(meta['image'], axis=0) for meta in batch]
    inputs = LLAVAOV_PREPROCESSOR(videos=video, text=text, return_tensors="pt", padding=True)
    return  {
        'inputs': inputs,
        'images': torch.stack([R18_PREPROCESSOR(Image.fromarray(meta['image'])) for meta in batch]).unsqueeze(1),
        'action': torch.stack([item['action'] for item in batch]),
        'action_mask': torch.stack([item['action_mask'] for item in batch])
    }

def infinite_shuffled_iterator(dataloader):
    
    while True:
        for item in dataloader:
            yield item

def create_OXE_datasets(
        batch_size, 
        action_chunk_length,
        use_recipe = 'UniAct-v1',   
        **kwargs):
    sample_weight_dict = {}
    dataloader_dict = {}
    if use_recipe not in OXE_NAMED_MIXTURES.keys():
        return sample_weight_dict, dataloader_dict
    for dataset_name, weight in OXE_NAMED_MIXTURES[use_recipe]:
        sample_weight_dict[dataset_name] = weight
        dataloader = DataLoader(OXEDataset(dataset_name,
                                            action_chunk_length=action_chunk_length),
                                            batch_size=batch_size, 
                                            num_workers=0, 
                                            pin_memory=True, 
                                            collate_fn=collate_fn)
        dataloader_dict[dataset_name] = infinite_shuffled_iterator(dataloader)
    return sample_weight_dict, dataloader_dict
