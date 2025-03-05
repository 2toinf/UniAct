
import torch
import numpy as np
import io
import h5py
import mmengine.fileio as fileio
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import pickle
from torch.utils.data import Dataset, DataLoader
from data.AIRData.config import AIRDATA_CONFIG
from data.AIRData.mixture import AIR_NAMED_MIXTURES
from data.utils import LLAVAOV_PREPROCESSOR, R18_PREPROCESSOR

class AIRDataset(Dataset):
    def __init__(self, 
                 dataset_name, 
                 action_chunk_length=1, 
                 action_normalize_way = 'min_max',
                 proprio_normalize_way = 'mean_std'):
        
        # init setting
        self.dataset_name = dataset_name
        self.config = AIRDATA_CONFIG[dataset_name]
        self.action_chunk_length = action_chunk_length
        self.action_normalize_way = action_normalize_way
        self.proprio_normalize_way = proprio_normalize_way
        # load meta files
        with open(self.config['meta_path'], "rb") as f: self.metas = pickle.load(f)
        for key, value in self.config['action_statics'].items():
            self.config['action_statics'][key] = np.array(value)
        
        if 'proprio_statics' in self.config.keys():
            for key, value in self.config['proprio_statics'].items():
                self.config['proprio_statics'][key] = np.array(value)
        # init id list for sample
        self.id_list = []
        for traj_id, item in enumerate(self.metas):
            self.id_list.extend([(traj_id, step_id) for step_id in range(item['length'])])
        
        # load image transform
        self.image_aug = transforms.Compose([
            transforms.RandomResizedCrop((384, 384), scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.1),
        ])

    def __len__(self):
        return len(self.id_list)

    def _get_multi_view_img(self, traj_id, step_id):
        img_np = []
        img_tensor = []
        for image_view in self.config['image']:
            img_path = fileio.join_path(
                self.config['image_path'], 
                self.metas[traj_id]['path'], 
                image_view,
                f"{self.config['img_prefix']}{step_id}{self.config['img_suffix']}")
            value = fileio.get(img_path)
            img_bytes = np.frombuffer(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as img: 
                img = img.convert('RGB')
            
            img = self.image_aug(img)
            if 'libero' in self.dataset_name:  
                img = TF.vflip(img)
                img = TF.hflip(img)
            img_np.append(np.asarray(img))
            img_tensor.append(R18_PREPROCESSOR(img))
        return {
            'img_np': np.stack(img_np),
            'img_tensor': torch.stack(img_tensor),
        }

    def _get_action(self, traj_id, step_id):
        # action chunking
        length = self.action_chunk_length
        action = self.metas[traj_id]['action'][step_id : step_id + length]
        if len(action) < length:
            padding_tensor = np.zeros_like(action[0])[np.newaxis, :].repeat(length - len(action), axis=0)
            action = np.concatenate([action, padding_tensor], axis=0)
        
        # action mask
        action_dim_mask = (action != 0.).sum(-1) != 0
        hidden_dim_mask = (self.config['action_statics']['99max'] - self.config['action_statics']['1min']) != 0
        action_mask = np.logical_and(np.expand_dims(action_dim_mask, axis=-1), hidden_dim_mask)
        
        # action normalize
        if self.action_normalize_way == 'min_max_99':
            action = (action - self.config['action_statics']['1min']) / \
                (self.config['action_statics']['99max'] - self.config['action_statics']['1min'])
        elif self.action_normalize_way == 'mean_std':
            action = (action - self.config['action_statics']['mean']) / self.config['action_statics']['std']
        elif self.action_normalize_way == 'min_max':
            action = (action - self.config['action_statics']['0min']) / \
                (self.config['action_statics']['100max'] - self.config['action_statics']['0min'])
            action = action * 2 - 1
        else:
            raise NotImplementedError
        
        return {'action': torch.from_numpy(action),
                'action_mask': torch.from_numpy(action_mask)}
        
    def _get_proprio(self, traj_id, step_id):

        if 'proprio_statics' not in self.config.keys(): return {}
        
        proprios = self.metas[traj_id]['proprios'][step_id]
        
        # proprios normalize
        if self.proprio_normalize_way == 'mean_std':
            proprios = (proprios - self.config['proprio_statics']['mean']) / self.config['proprio_statics']['std']
        else:
            raise NotImplementedError
        
        return {'proprios': torch.from_numpy(proprios)}

    def __getitem__(self, index):
        traj_id, step_id = self.id_list[index]
        return {
            'instruction': self.metas[traj_id]['instruction'],
            **self._get_multi_view_img(traj_id, step_id),
            **self._get_action(traj_id, step_id),
            **self._get_proprio(traj_id, step_id)
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

    video = [np.expand_dims(meta['img_np'][0], axis=0) for meta in batch]
    inputs = LLAVAOV_PREPROCESSOR(videos=video, text=text, return_tensors="pt", padding=True)
    data = {'inputs': inputs,
        'images': torch.stack([meta['img_tensor'] for meta in batch]),
        'action': torch.stack([item['action'] for item in batch]),
        'action_mask': torch.stack([item['action_mask'] for item in batch])
    }
    if 'proprios' in batch[0].keys():
        data['proprios'] = torch.stack([item['proprios'] for item in batch])
    return data

def infinite_shuffled_iterator(dataloader):
    epoch = 0
    while True:
        dataloader.sampler.set_epoch(epoch)
        for item in dataloader:
            yield item
        epoch += 1
            
def create_air_datasets(num_tasks,
                        global_rank,
                        batch_size,
                        action_chunk_length,
                        use_recipe='UniAct-1.0', 
                        **kwargs):
    sample_weight_dict = {}
    dataloader_dict = {}
    if use_recipe not in AIR_NAMED_MIXTURES.keys():
        return sample_weight_dict, dataloader_dict
    for dataset_name, weight in AIR_NAMED_MIXTURES[use_recipe]:
        sample_weight_dict[dataset_name] = weight
        dataset = AIRDataset(dataset_name = dataset_name,  action_chunk_length = action_chunk_length)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        dataloader = DataLoader(dataset, 
                        sampler=sampler,
                        batch_size=batch_size, 
                        num_workers=4, 
                        pin_memory=True, 
                        drop_last=True,
                        collate_fn=collate_fn)
        dataloader_dict[dataset_name] = infinite_shuffled_iterator(dataloader)
    return sample_weight_dict, dataloader_dict


    