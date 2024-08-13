
import torch
import time 
import cv2
import random
import datetime
import traceback
import os 
import sys
import json
from tqdm import tqdm 
import math
import PIL
import copy



from PIL import Image, ImageDraw, ImageFont
import numpy as np
# # Print current working directory
# print("Current working directory:", os.getcwd())

# # Add the current directory and its parent to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, current_dir)
# sys.path.insert(0, parent_dir)

# # Print sys.path
# print("Python sys.path in run_detection.py:")
# for path in sys.path:
#     print(path)

# # List contents of the modelsx/fast directory
# fast_dir = os.path.join(current_dir, 'modelsx', 'fast')
# if os.path.exists(fast_dir):
#     print("\nContents of modelsx/fast directory:")
#     for item in os.listdir(fast_dir):
#         print(item)

# # Now try to import
# try:
#     from modelsx.fast.custom_inference import FASTx
#     print("Successfully imported FASTx")
# except ImportError as e:
#     print(f"Import error: {e}")
#     import traceback
#     traceback.print_exc()


def recursive_add(path):
    if os.path.isdir(path):
        # print(path)
        sys.path.append(path)
        for dir_name in os.listdir(path):

            # print(os.path.join(path, dir_name))
            
            recursive_add(os.path.join(path, dir_name))

# for name in os.listdir(main_path):
#     print(os.path.join(main_path, name))

#     sys.path.append(os.path.join(main_path, name))
recursive_add(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelsx'))



from modelsx.fasterrcnn_inference import FasterRCNN
from modelsx.paddleocr.tools.infer.predict_rec import PaddleOCRx
from modelsx.fast.custom_inference import FASTx
# # Get the logger
from logger_setup import get_logger
logger = get_logger(__name__, "ocr.log", console_output=True)


# from models.create_colors import Colors  
# colors = Colors()  # create instance for 'from utils.plots import colors'

### ocr code 
class OCR():

    def __init__(self,params,logger,res_path="./results"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_detection_model_name = params["use_model"]
        self.text_recog_model_name = params["use_ocr_model"]
        print(f"\n\n self.text_detection_model_name:{self.text_detection_model_name}\n\n ")



        ### loading the model
        if self.text_detection_model_name == "fasterrcnn":
            self.model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=self.device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
            self.det_th = params["models"]["fasterrcnn"]["det_th"]
            self.det_box_type = "quad"
            print(f"FasterRCNN model created!!!")

        elif self.text_detection_model_name == "fast":
            self.model = FASTx(model_weights=params["models"]["fast"]["model_weights"], config=params["models"]["fast"]["config"],min_score=params["models"]["fast"]["min_score"],min_area=params["models"]["fast"]["min_area"],ema=params["models"]["fast"]["ema"])
            self.det_th = params["models"]["fast"]["det_th"]
            self.det_box_type = "poly"
            print(f"FAST model created!!!")
        else:
            self.model = None
        if self.text_recog_model_name == "paddleocr":
            print(f"__init__ OCR: initiating PaddleOCRx")
            self.ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"],rec_char_dict_path=params["ocr_models"]["paddleocr"]["rec_char_dict_path"])
            print(f"PaddleOCRx model created for text RECOG task!!!")

        else:
            self.ocr_model = None

        self.drop_score = 0.5
        self.logger = logger
        
        self.draw_img_save_dir =  res_path
        os.makedirs(self.draw_img_save_dir, exist_ok=True)

    def imgCrop(self,ori_im,dt_boxes):
        img_crop_list = []
        dt_boxes_list = []

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "quad":
                img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = self.get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
            dt_boxes_list.append(tmp_box)
        return img_crop_list,dt_boxes_list

    def sorted_boxes(self,dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def model_predcition(self,image,model):
        if self.text_detection_model_name =="fasterrcnn":
            dt_boxes, class_names, scores = model(image)
            dt_boxes = [self.increase_bbox_area_lengthwise(i) for i in dt_boxes]

            ### converting to paddle format
            dt_boxes =  self.convert_bbox_fasterrcnn2paddle(dt_boxes)
        elif self.text_detection_model_name == "fast":
            dt_boxes = model(image)
        else:
            print(f"[ERROR] Detection model not specified!!!")
            raise 
        print(f"\n\ndt_boxes:{dt_boxes} type:{type(dt_boxes)} shape:{dt_boxes.shape}\n\n")
        
        dt_boxes = self.sorted_boxes(dt_boxes)
        print(f"\n\ndt_boxes after sorted_boxes:{dt_boxes}\n\n")

        return dt_boxes


    def __call__(self,image,img_name=None, manualEntryx=None):
        st = time.time()
        org_img = image.copy()
        ### -------- TEXT DETECTION --------
        dt_boxes = self.model_predcition(image=image,model=self.model)
        self.logger.info(f"[INFO] time taken for text detection {time.time() - st } seconds")
        detected_texts = []
        detection_scores = []
        detected_bboxes = []

        print(f"org_img:{org_img.shape}")

        img_crop_list,detected_bboxes = self.imgCrop(ori_im=org_img,dt_boxes=dt_boxes)

        if self.ocr_model !=None:
            for cropped_image in img_crop_list:
                st = time.time() 
                cv2.imwrite(f"crop_{time.time()}.png",cropped_image)

                ocr_text,score = self.ocr_model(cropped_image)
                detected_texts.append(ocr_text)
                detection_scores.append(score)
                
                # print(f"[INFO] {datetime.datetime.now()}: time taken for text recognition {time.time() - st }  seconds")
                self.logger.info(f"[INFO] time taken for text recognition {time.time() - st } seconds x detected texts: {detected_texts} detection_scores:{detection_scores}")



        # ### plotting results and saving images 


        # ### saving output image
        # img_save_name = os.path.join(self.draw_img_save_dir, img_name[:-4] if img_name != None else str(self.img_count))+".png"
        # cv2.imwrite(
        #     img_save_name,
        #     draw_img[:, :, ::-1],
        # )
        # self.logger.debug(
        #     "The visualized image saved in {}".format(
        #         img_save_name
        #     )
        # )



        img_save_name ="TEST"
        
        return detected_texts,img_save_name

    def get_rotate_crop_image(self,img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img



    def get_minarea_rect_crop(self,img, points):
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img
    def get_rotate_crop_image(self,img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    
    
    def increase_bbox_area_lengthwise(self, bbox, factor=1.05):
        """
        Increase the bounding box area lengthwise by a given factor.
        
        Parameters:
        bbox (list): A bounding box in [x1, y1, x2, y2] format.
        factor (float): The factor by which to increase the length. Default is 1.1 (10% increase).
        
        Returns:
        list: The adjusted bounding box in [x1, y1, x2, y2] format.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        new_width = width * factor
        new_height = height * factor

        # Calculate new coordinates while keeping the center the same
        cx, cy = x1 + width / 2, y1 + height / 2
        new_x1 = max(cx - new_width / 2, 0)
        new_y1 = max(cy - new_height / 2, 0)
        new_x2 = max(cx + new_width / 2, 0)
        new_y2 = max(cy + new_height / 2, 0)

        return [new_x1, new_y1, new_x2, new_y2]

    def convert_bbox_fasterrcnn2paddle(self, bboxes):

        """
        Convert bounding boxes from [[x1, y1, x2, y2]] format to [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]] format,
        after increasing the area lengthwise by 10%.
        
        Parameters:
        bboxes (list): A list of bounding boxes, each in [x1, y1, x2, y2] format.
        
        Returns:
        list: A list of bounding boxes, each in [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]] format as numpy arrays with dtype float32.
        """
        converted_bboxes = []
        for bbox in bboxes:
            # increased_bbox = self.increase_bbox_area_lengthwise(bbox)
            x1, y1, x2, y2 = bbox
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            converted_bboxes.append(box)
        return np.array(converted_bboxes)







def main():
    try:
        # with open('./model_jsons/paramx.json', 'r') as f: ### docker 
        with open('./model_jsons/paramx_nodocker.json', 'r') as f: ### without docker 

            params = json.load(f)
        # Initialize the OCR model with the result path
        print(f"\n\n\n\n::::::::::::::::::::::\n\n\n")
        ocr_modelx = OCR(params,logger=logger,res_path=params["output_dir"])
    except:
        print(f"\n [ERROR] {datetime.datetime.now()} OCR model loading failed!!!\n ")
        traceback.print_exception(*sys.exc_info())
        sys.exit(1)
    
    image_dir = params["image_dir"]

    
    ### reading images and inferencing
    for im_name in tqdm(os.listdir(image_dir)):        
        img_path = os.path.join(image_dir, im_name)
        print(f"[INFO]{datetime.datetime.now()} working with img_path:{img_path}\n ")

        img = cv2.imread(img_path)
        res_txt, result_img_path = ocr_modelx(img,img_name=im_name)

if __name__ == '__main__':
    print(f"[INFO]{datetime.datetime.now()} ---------- PROCESS STARTED ----------\n ")
    main()
    print(f"[INFO]{datetime.datetime.now()} ---------- PROCESS COMPLETED ----------\n ")


