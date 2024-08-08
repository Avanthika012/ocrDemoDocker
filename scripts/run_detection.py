
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




from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.fasterrcnn_inference import FasterRCNN
from models.paddleocr.tools.infer.predict_rec import PaddleOCRx
# Get the logger
from logger_setup import get_logger
logger = get_logger(__name__, "ocr.log", console_output=True)


# from models.create_colors import Colors  
# colors = Colors()  # create instance for 'from utils.plots import colors'

### ocr code 
class OCR():

    def __init__(self,params,logger,res_path="./results"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        ### loading the model
        if params["use_model"] == "fasterrcnn":
            self.model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=self.device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
            self.det_th = params["models"]["fasterrcnn"]["det_th"]
            print(f"FasterRCNN model created!!!")

        else:
            self.model = None
        if params["use_ocr_model"] == "paddleocr":
            print(f"__init__ OCR: initiating PaddleOCRx")
            self.ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"],rec_char_dict_path=params["ocr_models"]["paddleocr"]["rec_char_dict_path"])
            print(f"PaddleOCRx model created for text RECOG task!!!")

        else:
            self.ocr_model = None

        self.drop_score = 0.5
        self.logger = logger
        
        self.draw_img_save_dir =  res_path
        os.makedirs(self.draw_img_save_dir, exist_ok=True)


    def __call__(self,image,img_name=None, manualEntryx=None):
        st = time.time()
        ### -------- TEXT DETECTION --------
        boxes, class_names, scores = self.model(image)
        # print(f"[INFO] {datetime.datetime.now()}: time taken for text detection {time.time() - st } seconds")
        self.logger.info(f"[INFO] time taken for text detection {time.time() - st } seconds x {len(class_names)} no. of texts detected!!!")
        detected_texts = []
        detection_scores = []
        detected_bboxes = []

        ### looping through all detected BBoxes or texts on an image
        for i in range(len(class_names)): 
            if scores[i]>=self.det_th: 
                # x1,y1,x2,y2 = boxes[i]
                x1,y1,x2,y2  = self.increase_bbox_area_lengthwise(boxes[i])
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cname= class_names[i]
                detected_bboxes.append([int(x1),int(y1),int(x2),int(y2) ])

                #### --------    OCR WORK    ----------------
                if self.ocr_model !=None:
                    cropped_image = image[y1:y2, x1:x2]

                    st = time.time() 
                    ocr_text,score = self.ocr_model(cropped_image)
                    detected_texts.append(ocr_text)
                    detection_scores.append(score)
                    
                    # print(f"[INFO] {datetime.datetime.now()}: time taken for text recognition {time.time() - st }  seconds")
                    self.logger.info(f"[INFO] time taken for text recognition {time.time() - st } seconds x detected texts: {detected_texts} detection_scores:{detection_scores}")

        ### plotting results and saving images 
        draw_img = self.draw_ocr_box_txt(
            image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            boxes=detected_bboxes,
            txts=detected_texts,
            scores=detection_scores,
            drop_score=self.drop_score,
            manualEntryx=manualEntryx
        )
        ### saving output image
        img_save_name = os.path.join(self.draw_img_save_dir, img_name[:-4] if img_name != None else str(self.img_count))+".png"
        cv2.imwrite(
            img_save_name,
            draw_img[:, :, ::-1],
        )
        self.logger.debug(
            "The visualized image saved in {}".format(
                img_save_name
            )
        )

        
        return detected_texts,img_save_name
    

    # def draw_ocr_box_txt(self,
    #     image,
    #     boxes,
    #     txts=None,
    #     scores=None,
    #     drop_score=0.5,
    #     font_path="./models/paddleocr/doc/fonts/simfang.ttf",
    #     font_size_factor=5,
    #     manualEntryx=None
    # ):
    #     # Validate the font_size_factor
    #     if not (0.5 <= font_size_factor <= 10.0):
    #         raise ValueError("font_size_factor should be between 0.5 and 10.0")

    #     h, w = image.height, image.width
    #     font = ImageFont.truetype(font_path, int(20 * font_size_factor))

    #     # Filter out texts based on scores
    #     valid_texts = [txt for idx, txt in enumerate(txts) if scores is None or scores[idx] >= drop_score]

    #     img_show = image.copy()

    #     if manualEntryx is not None:
    #         manual_entries = manualEntryx.strip().lower()  # Normalize to lower case
    #     else:
    #         manual_entries = None

    #     print(f"Drawing on canvas : valid_texts:{valid_texts}")

    #     pass_status = "Failed"
    #     status_color = (255, 0, 0)  # Red for failed

    #     if manual_entries is not None:
    #         for text in valid_texts:
    #             normalized_text = text.strip().lower()
    #             if any(char in manual_entries for char in normalized_text):
    #                 pass_status = "Passed"
    #                 status_color = (0, 255, 0)  # Green for passed
    #                 break

    #     # Calculate new height for the status
    #     line_height = 100  # Height of the line with some margin
    #     new_h = img_show.height + line_height + 40  # Add some margin at the top
    #     new_img = Image.new("RGB", (w, new_h), (0, 0, 0))  # Set background color to black
    #     new_img.paste(img_show, (0, 0))

    #     draw = ImageDraw.Draw(new_img)
    #     y_offset = img_show.height + 20  # Start drawing text below the current image with some margin

    #     # Draw the status text with color
    #     draw.text((10, y_offset), pass_status, fill=status_color, font=font)

    #     img_show = new_img  # Update the image with the newly created image

    #     return np.array(img_show)
    def draw_ocr_box_txt(self,
        image,
        boxes,
        txts=None,
        scores=None,
        drop_score=0.5,
        font_path="./models/paddleocr/doc/fonts/simfang.ttf",
        manualEntryx=None
    ):
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        boxes = self.convert_bbox_format(boxes) ### converting into an array with dtype=float32 for polygon drawing
        draw_left = ImageDraw.Draw(img_left)
        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            if scores is not None and scores[idx] < drop_score:
                continue
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            draw_left.polygon(box, fill=color)
            img_right_text = self.draw_box_txt_fine((w, h), box, txt, font_path)
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_right_text, [pts], True, color, 1)
            img_right = cv2.bitwise_and(img_right, img_right_text)
        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return np.array(img_show)
    
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

    def convert_bbox_format(self, bboxes):
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
        return converted_bboxes

    def draw_box_txt_fine(self,img_size, box, txt, font_path="./models/paddleocr/doc/fonts/simfang.ttf"):
        box_height = int(
            math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        )
        box_width = int(
            math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        )

        if box_height > 2 * box_width and box_height > 30:
            img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font = self.create_font(txt, (box_height, box_width), font_path)
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
            img_text = img_text.transpose(Image.ROTATE_270)
        else:
            img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font = self.create_font(txt, (box_width, box_height), font_path)
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

        pts1 = np.float32(
            [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
        )
        pts2 = np.array(box, dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        img_text = np.array(img_text, dtype=np.uint8)
        img_right_text = cv2.warpPerspective(
            img_text,
            M,
            img_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return img_right_text


    def create_font(self,txt, sz, font_path="./models/paddleocr/doc/fonts/simfang.ttf"):
        font_size = int(sz[1] * 0.99)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        if int(PIL.__version__.split(".")[0]) < 10:
            length = font.getsize(txt)[0]
        else:
            length = font.getlength(txt)

        if length > sz[0]:
            font_size = int(font_size * sz[0] / length)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        return font



def main():
    try:
        with open('./model_jsons/paramx.json', 'r') as f:
            params = json.load(f)
        # Initialize the OCR model with the result path
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


