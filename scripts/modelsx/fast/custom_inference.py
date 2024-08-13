import torch
import argparse
import os
import sys

# Get the directory of the current file (custom_inference.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (which should contain the dataset folder) to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import build_data_loader from dataset
from dataset import build_data_loader

# Print for debugging
print(f"Python sys.path in custom_inference.py:")
for path in sys.path:
    print(path)

import numpy as np 
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
import logging
import warnings
warnings.filterwarnings('ignore')
import json
from dataset.utils import scale_aligned_short, scale_aligned_long
from PIL import Image
import torchvision.transforms as transforms

class FASTx():
    def __init__(self,model_weights,config,min_score,min_area,ema, device="cuda",short_size=640) -> None:
        self.cfg = Config.fromfile(config)
        self.checkpoint = model_weights
        self.ema = ema
        self.short_size= short_size
        self.device = device

        if min_score is not None:
            self.cfg.test_cfg.min_score = min_score
        if min_area is not None:
            self.cfg.test_cfg.min_area = min_area

        ### loading model
        self.model = self.create_model(device=device)

    
    def create_model(self,device):
        model = build_model(self.cfg.model)

        if device=="cpu":
            model.cpu()
        else:
            model.cuda()

        if self.checkpoint is not None:
            if os.path.isfile(self.checkpoint):
                print("Loading model and optimizer from checkpoint '{}'".format(self.checkpoint))
                logging.info("Loading model and optimizer from checkpoint '{}'".format(self.checkpoint))
                sys.stdout.flush()
                checkpoint = torch.load(self.checkpoint)
                
                if not self.ema:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint['ema']

                d = dict()
                for key, value in state_dict.items():
                    tmp = key.replace("module.", "")
                    d[tmp] = value
                model.load_state_dict(d)
            else:
                print("No checkpoint found at '{}'".format(args.checkpoint))
                raise
    
        model = rep_model_convert(model)
        # fuse conv and bn
        model = fuse_module(model)

        ### eval mode on 
        model.eval()
        return model 
    def preProcess(self,img):
        img = img[:, :, [2, 1, 0]]
        img_meta = dict(
            org_img_size=torch.tensor(np.array(img.shape[:2])).unsqueeze(0)
        )
        print(f"\n\n org_img_size:{torch.tensor(np.array(img.shape[:2])).unsqueeze(0)}")

        img = scale_aligned_short(img, self.short_size)

        img_meta.update(dict(
            img_size=torch.tensor(np.array(img.shape[:2])).unsqueeze(0)
        ))

        print(f"img_size:{torch.tensor(np.array(img.shape[:2])).unsqueeze(0)}\n\n ")

        ### transformations
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img = torch.unsqueeze(img,0)
        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        if self.device=="cuda":
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        
        data.update(dict(cfg=self.cfg))
        
        return data

    def convert_fast2paddle_res(self, arr):
        # Group the coordinates into pairs
        grouped = [[arr[i], arr[i+1]] for i in range(0, len(arr), 2)]
        
        # Convert the grouped list to a numpy array
        np_arr = np.array(grouped)
        
        # Add two extra dimensions to make it (1, 16, 2)
        final_arr = np_arr[np.newaxis, ...]
        
        return final_arr

    
    def forward(self,data):
        # running inference 
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs

    def postProcess(self,outputs):
        results = outputs['results']
        dt_boxes = results[0]["bboxes"]
        dt_scores = results[0]["scores"]
        ### code to be added for bbox filtering based on scores or confidence th

        ###coverting dt_boxes for for paddle integration
        print(dt_boxes)
        dt_boxes = self.convert_fast2paddle_res(dt_boxes[0])

        return dt_boxes

    # A Method that mergers everything together
    def __call__(self, image):

        data = self.preProcess(image)
        print(f"\n\n\n imgs:{len(data['imgs'])} {data['imgs'].shape} {data['imgs'].size()} {type(data['imgs'])} {data['imgs'].dtype}\n\n")

        outputs = self.forward(data)
        dt_boxes = self.postProcess(outputs)
        return dt_boxes

