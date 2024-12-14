""" Detect people wearing masks in videos
"""
from pathlib import Path

import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from common.facedetector import FaceDetector
from train import MaskDetector

from ultralytics import YOLO

# Load a model

# Run batched inference on a list of images

# Process results list


@torch.no_grad()
def tagVideo(videopath):
    """ detect if persons in video are wearing masks or not
    """
    mask_model = MaskDetector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_model.load_state_dict(torch.load('models/face_mask.ckpt', map_location=device)['state_dict'],
                          strict=False)
    
    mask_model = mask_model.to(device)
    mask_model.eval()
    action_model = YOLO("best.pt")  # pretrained YOLO11n model
    
    faceDetector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    cap = cv2.VideoCapture(0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]
    state, image = cap.read()
    while True:
        state, frame = cap.read()
        src_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        action_results = action_model(frame)  # return a list of Results objects
        """
        draw body rect
        """
        for b in action_results:
            body_pos = b.boxes.xyxy.cpu().numpy()
            if len(body_pos) > 0:
                for body in body_pos:
                    classes = b.boxes.cls
                    print(classes)
                    x1,y1,x2,y2 = [int(t) for t in body]
                    cv2.rectangle(src_frame, (x1, y1), (x2, y2), (128, 0, 0), 2)

        """
        face detect
        """
        faces = faceDetector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face
            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            # predict mask label on extracted face
            faceImg = src_frame[yStart:yStart+height, xStart:xStart+width]
            output = mask_model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)
            if predicted == 1:
                for b in action_results:
                    body_pos = b.boxes.xyxy.cpu().numpy()
                    clss = b.boxes.cls.tolist()
                    for body, cls in zip(body_pos, clss):
                        x1, y1, x2, y2 = [int(t) for t in body]
                        if x1 < xStart and y1 < yStart and x2 > xStart + width and y2 > yStart + height:
                            #print(b.boxes.cls)
                            stat = 'abnormal'
                            if int(cls) == 1:
                                stat = "inductor"
                            cv2.putText(src_frame,
                                        stat,
                                        (x1, y1 - 20),
                                        font, 1, labelColor[0], 2)
            else:
                for b in action_results:
                    body_pos = b.boxes.xyxy.cpu().numpy()
                    for body in body_pos:
                        x1, y1, x2, y2 = [int(t) for t in body]
                        cv2.putText(src_frame,
                                        'normal',
                                        (x1, y1 - 20),
                                        font, 1, labelColor[1], 2)
            # draw face frame
            cv2.rectangle(src_frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)
            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2

            # draw prediction label
            cv2.putText(src_frame,
                        labels[predicted],
                        (textX, yStart-20),
                        font, 1, labelColor[predicted], 2)
        cv2.imshow('main', src_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagVideo(0)
