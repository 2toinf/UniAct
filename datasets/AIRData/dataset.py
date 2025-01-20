
from transformers import LlavaOnevisionProcessor
import torch
import numpy as np
import io
import mmengine.fileio as fileio
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import pickle
from torch.utils.data import Dataset, DataLoader
from datasets.AIRData.config import AIRDATA_CONFIG
from datasets.AIRData.mixture import AIR_NAMED_MIXTURES
LLAVAOV_PREPROCESSOR = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf",local_files_only = True)

class AIRDataset(Dataset):
    def __init__(self, dataset_name, action_chunk_length=1, action_normalize_way = 'min_max_99'):
        
        # init setting
        self.dataset_name = dataset_name
        self.config = AIRDATA_CONFIG[dataset_name]
        self.action_chunk_length = action_chunk_length
        self.action_normalize_way = action_normalize_way
        
        # load meta files
        with open(self.config['meta_path'], "rb") as f: self.metas = pickle.load(f)
        for key, value in self.config['action_statics'].items():
            self.config['action_statics'][key] = np.array(value)
        
        # init id list for sample
        self.id_list = []
        for traj_id, item in enumerate(self.metas):
            self.id_list.extend([(traj_id, step_id) for step_id in range(item['length'])])
        
        # load image transform
        self.image_aug = transforms.Compose([
            transforms.RandomResizedCrop((384, 384), scale=(0.7, 0.9), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05),
        ])

    def __len__(self):
        return len(self.id_list)

    def _get_single_img(self, traj_id, step_id):
        img_path = fileio.join_path(
            self.config['image_path'], 
            self.metas[traj_id]['path'], 
            self.config['image']['primary'],
            f"{self.config['img_prefix']}{step_id}{self.config['img_suffix']}")
        value = fileio.get(img_path)
        img_bytes = np.frombuffer(value, np.uint8)
        buff = io.BytesIO(img_bytes)
        with Image.open(buff) as img: 
            img = img.convert('RGB')
        if self.dataset_name == 'libero':  img = TF.vflip(img)
        return np.asarray(img)

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
                (self.config['action_statics']['99max'] - self.config['action_statics']['1min'] + 1e-6)
        else:
            raise NotImplementedError
        
        return {'action': torch.from_numpy(action),
                'action_mask': torch.from_numpy(action_mask)}

    def __getitem__(self, index):
        traj_id, step_id = self.id_list[index]
        return {
            'instruction': self.metas[traj_id]['instruction'],
            'image': self._get_single_img(traj_id, step_id),
            **self._get_action(traj_id, step_id)
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
        'action': torch.stack([item['action'] for item in batch]),
        'action_mask': torch.stack([item['action_mask'] for item in batch])
    }

def infinite_shuffled_iterator(dataset_name, action_chunk_length=4, batch_size=2):
    dataloader = DataLoader(AIRDataset(dataset_name = dataset_name,  action_chunk_length = action_chunk_length), 
                        batch_size=batch_size, 
                        num_workers=2, 
                        pin_memory=True, 
                        drop_last=True,
                        shuffle=True,
                        collate_fn=collate_fn)
    while True:
        for item in dataloader:
            yield item
            
def create_AIR_datasets(batch_size,
                        action_chunk_length,
                        use_recipe='uniact-1.0version', **kwargs):
    sample_weight_dict = {}
    dataloader_dict = {}
    for dataset_name, weight in AIR_NAMED_MIXTURES[use_recipe]:
        sample_weight_dict[dataset_name] = weight
        dataloader_dict[dataset_name] = infinite_shuffled_iterator(dataset_name,
                                                   batch_size=batch_size,
                                                   action_chunk_length=action_chunk_length)
    return sample_weight_dict, dataloader_dict


    