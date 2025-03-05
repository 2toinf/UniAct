import h5py
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

data_names = ['libero_90']

for data_name in data_names:
    obs_keys = ['agentview_rgb', 'eye_in_hand_rgb', 'joint_states', 'gripper_states']
    base_dir = f'../{data_name}'
    save_base_dir = f'../dataset/libero/data_jpg/{data_name}'
    hdf5_path_list = os.listdir(base_dir)

    for hdf5_path in hdf5_path_list:
        # open file
        hdf5_file = h5py.File(os.path.join(base_dir, hdf5_path), 'r', swmr=False, libver='latest')
        demos = list(hdf5_file["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos_sorted = [demos[i] for i in inds]
        print(demos)
        print(demos_sorted)

        # language instruction
        lang_instruction = hdf5_path.split("_demo.hdf5")[0].replace("_", " ")
        lang_instruction = ''.join([char for char in lang_instruction if not (char.isupper() or char.isdigit())])
        lang_instruction = lang_instruction.lstrip(' ')
        print(lang_instruction)


        # get data
        all_data = dict()
        for ep in tqdm(demos_sorted):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in obs_keys}
            all_data[ep]["action"] = hdf5_file["data/{}/actions/".format(ep)][()].astype('float32')
            all_data[ep]['state'] = hdf5_file["data/{}/robot_states/".format(ep)][()].astype('float32')
            all_data[ep]["attrs"]['lang'] = lang_instruction
            
            # save
            save_dir = os.path.join(save_base_dir, hdf5_path.split('.')[0])
            save_dir = os.path.join(save_dir, ep)
            os.makedirs(f"{save_dir}/image0", exist_ok=True)
            os.makedirs(f"{save_dir}/image1", exist_ok=True)
            
            # save action
            action_path = f"{save_dir}/action.npy"
            state_path = f"{save_dir}/state.npy"
            action = all_data[ep]["action"]
            state = all_data[ep]["state"]
            np.save(action_path, action)
            np.save(state_path, state)
            
            # save lang
            with open(f"{save_dir}/lang.txt", "w") as f:
                f.write(lang_instruction)
            
            # save image
            for idx in range(all_data[ep]["attrs"]["num_samples"]):
                third_image = Image.fromarray(all_data[ep]["obs"]['agentview_rgb'][idx].astype(np.uint8))
                wrist_image = Image.fromarray(all_data[ep]["obs"]['eye_in_hand_rgb'][idx].astype(np.uint8))
                
                third_image_path = f"{save_dir}/image0/{idx}.jpg"
                wrist_image_path = f"{save_dir}/image1/{idx}.jpg"
                
                third_image.save(third_image_path)
                wrist_image.save(wrist_image_path)