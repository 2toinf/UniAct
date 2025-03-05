
from transformers import LlavaOnevisionProcessor
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torch

LLAVAOV_PREPROCESSOR = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf",local_files_only = False)
R18_PREPROCESSOR = transforms.Compose(
    [
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
    ]
)