import os
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
from local_paths import get_local_image_path, resolve_existing_path


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path="a.jsonl", prompt_keys = ['short_en', 'detailed_en', 'short_zh', 'detailed_zh'], num_prompts=16,have_gt=False, data_root: str | Path | None = None):
        # only read the first num_prompts lines
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else Path(__file__).resolve().parent
        self.dataset_path = resolve_existing_path(dataset_path, self.data_root)
        with open(self.dataset_path, 'r') as f:
            all_data = [json.loads(line.strip()) for line in f.readlines()]
        self.prompts = []
        self.images_path = []
        self.have_gt_image = have_gt
        for data in all_data:
            selected_keys =  random.choice(prompt_keys)
            prompt = data[selected_keys]
            self.prompts.append(prompt)
            if have_gt:
                image_path = get_local_image_path(data, jsonl_path=self.dataset_path, data_root=self.data_root)
                self.images_path.append(image_path)
            if len(self.prompts) >= num_prompts:
                break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if self.have_gt_image:
            return  self.prompts[idx], self.images_path[idx]
        else:
            return  self.prompts[idx], {}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        return prompts
