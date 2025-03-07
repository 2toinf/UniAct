# [CVPR 2025] Universal Actions for Enhanced Embodied Foundation Models
[[Project Page](https://2toinf.github.io/UniAct/)] [[Paper](https://arxiv.org/abs/2501.10105)]

[Jinliang Zheng](https://2toinf.github.io/)\*, [Jianxiong Li](https://Facebear-ljx.github.io/)\*, Dongxiu Liu\*,  [Yinan Zheng](https://zhengyinan-air.github.io/), [Zhihao Wang](https://zh1hao.wang/Robo_MUTUAL/), Zhonghong Ou, [Yu Liu](https://liuyu.us/), [Jingjing Liu](https://air.tsinghua.edu.cn/en/info/1046/1194.htm), [Ya-Qin Zhang](https://air.tsinghua.edu.cn/en/info/1046/1188.htm), [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/), 


||||||
|--|--|--|--|--|
| <img width="140"  src="static/videos/widowx/1.gif"/> | <img width="140"  src="static/videos/widowx/7.gif"/> |<img width="140"  src="static/videos/widowx/6.gif"/>|<img width="140"  src="static/videos/widowx/11.gif"/> | <img width="140"  src="static/videos/widowx/4.gif"/>|
| <img width="140"  src="static/videos/libero/1.gif"/> | <img width="140"  src="static/videos/libero/2.gif"/> |<img width="140"  src="static/videos/libero/3.gif"/>|<img width="140"  src="static/videos/libero/4.gif"/> | <img width="140"  src="static/videos/libero/5.gif"/>|
| <img width="140"  src="static/videos/airbot/1.gif"/> | <img width="140"  src="static/videos/airbot/2.gif"/> |<img width="140"  src="static/videos/airbot/3.gif"/>|<img width="140"  src="static/videos/airbot/4.gif"/> | <img width="140"  src="static/videos/airbot/5.gif"/>|



## Introduction

we introduce UniAct, a new embodied foundation modeling framework operating in the Universal Action Space. Our learned universal actions capture the generic atomic behaviors across diverse robots by exploiting their shared structural features, and enable enhanced cross-domain data utilization and cross-embodiment generalizations by eliminating the notorious heterogeneity. Moreover, the universal actions can be efficiently translated back to heterogeneous actionable commands by simply adding embodiment-specific details, from which fast adaptation to new robots becomes simple and straightforward. Our 0.5B instantiation of UniAct outperforms 14X larger SOTA embodied foundations models in extensive evaluations on various real-world and simu- lation robotic environments, showcasing exceptional cross-embodiment control and adaptation capability, highlighting the crucial benefit of adopting universal actions

<img width="1000"  src="static/images/intro_final.jpg">


## Citation & Contact
- If you find this repo useful, please kindly cite us:
```
@misc{zheng2025universalactionsenhancedembodied,
      title={Universal Actions for Enhanced Embodied Foundation Models}, 
      author={Jinliang Zheng and Jianxiong Li and Dongxiu Liu and Yinan Zheng and Zhihao Wang and Zhonghong Ou and Yu Liu and Jingjing Liu and Ya-Qin Zhang and Xianyuan Zhan},
      year={2025},
      eprint={2501.10105},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.10105}, 
}
```
- If you have any questions about the code, feel free to raise an issue or contact the author directly: [Jinliang Zheng](https://2toinf.github.io/), [Jianxiong Li](https://facebear-ljx.github.io/)


## Quick Start 

Please note that the following guidance is only for model deployment, please kindly refer to [Train](#training-guidance)


### Install Package and Requirements

```bash
git clone https://github.com/2toinf/UniAct-Preview.git
cd UniAct-Preview
pip install -r requirements.txt
```

### Load models

Firstly download the pretrained UniAct from [Model Zoo](#model-zoo)

```python
import models.UniAct_V1
from timm.models import create_model
uniact_model = create_model("UniAct_05B_CodeBook_256_V1")

### firstly load the pretrained universal action extractor / vision backbone / codebook
uniact_model.load_state_dict(torch.load("model path here"), strict=False)

### Then load embodiment-specifc decoders
uniact_model.load_state_dict(torch.load("decoder path here"), strict=False)
```


### Prepare the input


```python

from datasets.utils import LLAVAOV_PREPROCESSOR, R18_PREPROCESSOR
from PIL import Image

proprios = None # defaultly disable the proprios, use it for ACT decoder(Need to be normalized)
language_instruction = "your language instruction here"
image_list = ["image-view1 path here", "image-view2 path here"]
img_np = []
img_tensor = []
for image in image_list:
    with Image.open(image) as img: 
        img = img.convert('RGB')
    img_np.append(np.asarray(img))
    img_tensor.append(R18_PREPROCESSOR(img))

img_np = np.stack(img_np),
img_tensor = torch.stack(img_tensor),


text = [LLAVAOV_PREPROCESSOR.apply_chat_template([
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text":  language_instruction},
                ]
            }], add_generation_prompt=True)]

video = [np.expand_dims(img_np[0], axis=0)] # only use the primary view for extractor!
inputs = LLAVAOV_PREPROCESSOR(videos=video, text=text, return_tensors="pt", padding=True)


inputs = {'inputs': inputs.to('cuda', torch.bfloat16),
        'images': img_tensor.unsqueeze(0).to('cuda', torch.bfloat16),
    }
if proprios is not None: inputs['proprios'] = proprios.to('cuda', torch.bfloat16)

```


### Infer the model

```python
pred_action = uniact_model.infer(
    domain_name = "libero-1-rgb", # check the model_config.py for the domain_name
    **inputs
)

```
**Note:** Please remember to denormalize the 'pred_action', kindly check the action statics for [AIRData](data/AIRData/config.py) and [OXE](data/OXE/action_statics.py)

## Model Zoo

| Models    | Description  | ckpt |  Action normalize method| Observation type | Avg Succ Rate | 
| --------- | ------------------- |  ------------------- | ---------|-------------------------|-------------------------------------------------------- |
| Basemodel | Params for Universal Action Extractor / Vision Backbone / Universal Action Codebook | [hf_link](https://huggingface.co/2toINF/UniAct/blob/main/basemodel.pt) | - | Static view | - |
| Libero-MLP-Decoder | Params for MLP decoder on Libero  | [hf_link](https://huggingface.co/2toINF/UniAct/blob/main/libero_mlp.pt)    | mean-std | Static view | 61.3% |
| Bridge-MLP-Decoder | Params for MLP decoder on Bridge  |  [hf_link](https://huggingface.co/2toINF/UniAct/blob/main/bridge_mlp.pt)   | mean-std | Static view| 63.3% |

**As we haven't access the performance of other decoder heads, we will not release them. If you have any questions about this, please feel free to contact us.**

## Evaluation on Libero

### Installation
Please follow the guide in the official repo to install the [LIBERO simulation](https://github.com/Lifelong-Robot-Learning/LIBERO).

### Reproduce the results
1. **LIBERO (MLP Head)**

You can directly run the following command by replacing `YOUR_BASEMODEL_CKPT_PATH` and `YOUR_HEAD_CKPT_PATH` as your base model and head ckpt pathes, e.g., `/data/UniAct/basemodel.pt` and `/data/UniAct/libero_mlp.pt`:

```
python eval/libero/run_uniact_libero_eval.py \
    --base_path YOUR_BASEMODEL_CKPT_PATH \
    --head_path YOUR_HEAD_CKPT_PATH \
    --num_episodes 20 \
```

1. **LIBERO (ACT Head)**

You can directly run the following command by replacing `YOUR_BASEMODEL_CKPT_PATH` and `YOUR_HEAD_CKPT_PATH` as your base model and head ckpt pathes, e.g., `/data/UniAct/basemodel.pt` and `/data/UniAct/ACT_Libero.pt`:

```
python eval/acthead_libero/run_uniact_libero_eval.py \
    --base_path YOUR_BASEMODEL_CKPT_PATH \
    --head_path YOUR_HEAD_CKPT_PATH \
    --num_episodes 20 \
```

## Training Guidance

Firstly install the required packages for training, kindly refer to [train/requirements.txt](train/requirements.txt)

#### 1. Prepare the Data

1. Firstly download OXE dataset(tfds files) from the official [repo](https://robotics-transformer-x.github.io/) 

2. Fill the file path in [dataset.py](datasets/OXE/dataset.py)

```python
# set this if you store the files in s3 ceph
S3Path = ''
# set this if you store the files in local machine
LOCAL_OXE = ''
```

#### 2. Prepare the model

**As we have refined the data and there may be some conflict to your own data. Please carefully fill the model_config:**
Fill the model settings in [model_config.py](models/model_config.py).
Currently support decoders:
- ACT decoder (Refer to [ACT_decoder.py](decoders/ACT_decoder.py) for specifc name)
- MLP decoder (Refer to [MLP_decoder.py](decoders/MLP_decoder.py) for specifc name)


#### 3. Run the following script

```bash
srun -p mozi-S1 -n8  --gres=gpu:8  --ntasks-per-node=8 \
    python -u train/slurm_deepspeed_train.py \
        --model UniAct_05B_CodeBook_256_V1_Pretrain \
        --recipe oxe_magic_soup \
        --iters 1000000 \
        --start_iters 0 \
        --initial_t 2.0 \
        --final_t 0.5 \
        --batch-size 32 \
        --lr 1e-5 \
        --grad_accumulation_steps 1 \
        --output_dir exp/pretrain \
        --save_interval 10000 \
        --precision bf16 
```



### Fast-Adapt to your embodiment
We recommend you to use 'AIRData' as the data engine to train UniAct on your own embodiment. And that is what we do when we train UniAct on Libero! It may require some data reconstruction on your own dataset. We provide the data processing script for Libero as an example. Kindly refer to [Libero_hdf52jpg.py](datasets/AIRData/Preprocess/Libero_hdf52jpg.py). The image files structures should be as follow:

```
|-- traj-1
|  `-- frame-0.jpg
|  `-- frame-1.jpg
|  `-- frame-2.jpg
|  `-- ...
|  `-- frame-41.jpg
|-- traj-2
|  `-- frame-0.jpg
|  `-- frame-1.jpg
|  `-- frame-2.jpg
|  `-- ...
|  `-- frame-46.jpg
...
```

#### Data reconstruction

After you transfer the data into jpg format,  you need to follow the following instructions to adapt it to the codebase:

1. Construct a meta file(.pkl) as following structures:
. Here is a example: [Libero.pkl](https://huggingface.co/2toINF/UniAct/blob/main/libero_meta_file.pkl)

```
|-- 
|  `-- path: 'traj-1'
|  `-- length: 41   
|  `-- instruction: 'pick up the red cup'
|  `-- action: np.ndarray with shape(41, dim_action)
|  `-- proprios: np.ndarray with shape(41, dim_proprio)
|-- 
|  `-- path: 'traj-2'
|  `-- length: 46   
|  `-- instruction: '...'
|  `-- action: np.ndarray with shape(46, dim_action)
|  `-- proprios: np.ndarray with shape(46, dim_proprio)
|...
```



2. Modify the [config.py](datasets/AIRData/config.py)
please add your data meta infos follow the data structures in the file


3. Modify the [mixture.py](datasets/AIRData/mixture.py)
    
    
4. Modify the [model_config.py](models/model_config.py)
Please choose one decoder head for your embodiment and revise the file, currently support:
    - [MLP decoder](models/decoders/MLP_decoder.py)
    - [ACT decoder](models/decoders/ACT_decoder.py) 

#### Run the following script

```bash
srun -p mozi-S1 -n8  --gres=gpu:8  --ntasks-per-node=8   \
    python -u slurm_deepspeed_train.py \
        --model UniAct_05B_CodeBook_256_V1_For_Fast_Adaptation \
        --recipe ACT-Libero \
        --iters 1000000 \
        --start_iters 0 \
        --batch-size 32 \
        --lr 1e-4 \
        --grad_accumulation_steps 1 \
        --output_dir exp/Libero \
        --save_interval 10000 \
        --port 12345 \
        --seed 178945 \
        --precision bf16 \
        --resume "path of pretrained basemodel"
```


### Full-train UniAct on data-recipe in the paper
1. prepare the OXE dataset and AIRData following the above instruction
2. set `UniAct-1.0` as the data-recipe in the script
3. set `UniAct_05B_CodeBook_256_V1_Pretrain` as the model in the script

📆 TODO

- [X] Release training codebase.
- [X] Release code for depolyment.
- [X] Release model checkpoints.
- [X] Release training guidance.


## Acknowledgement

This work is built upon the [huggingface](https://github.com/huggingface/transformers.git) and [llava-one-vision](https://github.com/LLaVA-VL/LLaVA-NeXT.git).



