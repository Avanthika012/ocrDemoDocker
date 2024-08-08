import torch
import cv2, os
import numpy as np
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from models.resizing import ResizeShortestEdge
from detectron2.config import get_cfg
import datetime


class FasterRCNN:
    """
    This class defines an interface for running object detection using a pre-loaded Faster RCNN model.
    
    Attributes:
        cfg (object): The configuration object for the Faster RCNN model.
        model (object): The pre-loaded Faster RCNN model.
        classes (list): The list of classes the model can detect.
        aug (object): The augmentation transform to be applied on input images.
    """
    def __init__(self, model_weights,detection_thr, classes,config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",  device="cuda"):
        """
        Initializes the FasterRCNN class.

        Args:
            model_weights (str): Path to the model weights.
            config_path (str): Path to the configuration file.
            classes (list): List of class names the model can detect.
            score_thresh (float): Confidence threshold for inference.
            device (str): Device to run inference ("cuda" or "cpu").
        """
        # Preparing config
        self.cfg = self.editCfg(model_weights=model_weights, config_path=config_path, classes=classes,detection_thr=detection_thr)
        # Loading model
        self.model = self.load_model()
        # Putting model to GPU
        self.to(device)
        # Loading Aug function
        self.aug = ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)
        
        print(f"[INFO] {datetime.datetime.now()}: FasterRCNN loaded!!!\n")

    # Function to load the cfg file and make changes to it 
    # Depending on the params passed to it
    def editCfg(self, model_weights, config_path, classes,detection_thr):
        """
        Edits the configuration parameters for the model.

        Args:
            model_weights (str): Path to the model weights.
            config_path (str): Path to the config file.
            classes (list): List of classes the model can detect.
            score_thresh (float): Confidence threshold for inference.

        Returns:
            object: The updated configuration object.
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        # cfg.merge_from_file(params["params"]["config_path"]) 
        cfg.MODEL.WEIGHTS = model_weights   # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(detection_thr)  # set a custom testing threshold
        self.classes = classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes)
        return cfg
    
    # Function to load the model 
    # Based on the cfg file loaded
    def load_model(self):
        """
        Loads the Faster RCNN model based on the provided configuration.

        Returns:
            object: The loaded model.
        """
        model = build_model(self.cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        return model

    # Function that does the proper processing 
    # And converting to the proper format for model inferecing
    def preProcess(self, image):
        """
        Preprocesses the input image for inference.

        Args:
            image (array): The input image.

        Returns:
            dict: Dictionary containing the processed image and its dimensions.
        """
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        if self.input_format=='RGB':
                self.np_image = self.np_image[:,:,::-1]
        height,width = image.shape[:2]
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs

    # Passing the inputs to the model and doing the inference
    def forward(self, input):
        """
        Runs the preprocessed image through the model.

        Args:
            input (dict): Dictionary containing the processed image and its dimensions.

        Returns:
            dict: The model's raw output.
        """
        with torch.no_grad():
            predictions = self.model([input])[0]
        return predictions
    
    # Function to do the postprocessing and return output in proper format
    def postProcess(self, pred_output):
        """
        Post-processes the model's raw output.

        Args:
            pred_output (dict): The model's raw output.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        pred_output = pred_output["instances"]
        boxes = pred_output.pred_boxes
        pred_boxes_array = boxes.tensor.cpu().numpy().astype("int").tolist()
        classes = pred_output.pred_classes.cpu().numpy().tolist()
        scores = pred_output.scores.cpu().numpy().tolist()
        class_names = [self.classes[class_ind] for class_ind in classes]
        return pred_boxes_array, class_names, scores
    
    # Method to put model to specific device
    def to(self, device="cuda"):
        """
        Moves the model to the specified device.

        Args:
            device (str): Device to move the model to ("cuda" or "cpu").
        """
        if device=="cpu":
            self.model.cpu()
        else:
            self.model.cuda()
    
    # A Method that mergers everything together
    def __call__(self, image):
        """
        Makes the class callable for easier inference.

        Args:
            image (array): The input image.

        Returns:
            tuple: Bounding boxes, class names, and confidence scores of detected objects.
        """
        input = self.preProcess(image=image)
        pred_output = self.forward(input=input)
        result = self.postProcess(pred_output=pred_output)
        boxes, class_names, scores = result
        return boxes, class_names, scores