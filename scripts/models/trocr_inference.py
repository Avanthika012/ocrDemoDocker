import torch
import numpy as np
import glob as glob
import datetime
 
 
from tqdm.notebook import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
seed_everything(42)

@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = 'microsoft/trocr-base-printed'
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class TROCR():
    def __init__(self,model_weights,MODEL_NAME='microsoft/trocr-base-printed' ) -> None:
        self.MODEL_NAME = MODEL_NAME
        self.model_weights = model_weights # Model weights


        self.processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
        self.trained_model = VisionEncoderDecoderModel.from_pretrained(self.model_weights).to(device)

        print(f"[INFO] {datetime.datetime.now()}: TROCR loaded!!!\n")


    def ocr(self, image, processor, model):
        """
        :param image: PIL Image.
        :param processor: Huggingface OCR processor.
        :param model: Huggingface OCR model.
    
    
        Returns:
            generated_text: the OCR'd text string.
        """
        # We can directly perform OCR on cropped images.
        pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    # Method to call for direct inferencing
    def __call__(self, image):
        """
        Makes the class callable. Simplifies the process of running an image through the entire pipeline.

        Args:
            image (array): The input image.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        text = self.ocr(image, self.processor, self.trained_model)

        return text