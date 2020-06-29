# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)
small_model_path = os.path.join('..', 'snapshots', 'small_05_15.h5')
model_path_p = os.path.join('..', 'snapshots', 'resnet50_csv_11.h5')

# load retinanet model
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

# load retinanet model
model = models.load_model(model_path_p, backbone_name='resnet50')
small_model = models.load_model(small_model_path, backbone_name='resnet50')
small_model = models.convert_model(small_model)

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names_p = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
labels_to_names = {0: 'ball'}


class BoxIntersect():
    def __init__(self):
        self.boxes = []

    def add_box(self, box):
        for idx, b in enumerate(self.boxes):
            if (box[0] >= b[0] and box[0] <= b[2]) or (b[0] > box[0] and b[0] < box[2]):
                if (box[1] >= b[1] and box[1] <= b[3]) or (b[1] > box[1] and b[1] < box[3]):
                    new_box = (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
                    self.boxes[idx] = new_box
                    break
        else:
            self.boxes.append(box)
    def total_area(self):
        tot_area = 0
        for b in self.boxes:
            area = (b[2] - b[0]) * (b[3] - b[1])
            tot_area += area

        return tot_area

cap = cv2.VideoCapture('./IMG_0045.MOV')
x = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("out.mp4", fourcc,30, (1920, 1080));
filt=True
csv_file = open("csvfile_45", "w")
while cap.isOpened():

  # Capture frame-by-frame
    ret, image = cap.read()
    image = cv2.rotate(image, cv2.ROTATE_180)
    if not ret: print("bad");break
    print(x)
    x += 1
    #if x > 300:
    #    break
    # copy to draw on
    #draw = image.copy()
    draw2 = image.copy()
    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    print(image.shape)
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    # visualize detections
    bi = BoxIntersect()
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if label != 0:continue
        if score < 0.9:
            break

        color = label_color(label)

        b = box.astype(int)
        print("score")
        bfinal=None
        if filt:
            mid_x = int((b[2] - b[0]) / 2 + b[0])
            mid_y = int((b[3] - b[1]) / 2 + b[1])
        
            x_min = mid_x - 75
            x_max = mid_x + 75
            y_min = mid_y - 75
            y_max = mid_y + 75
            
            cropped = draw2[y_min:y_max,x_min:x_max].copy()
            cropped = preprocess_image(cropped)
            try:
               cropped, scale2 = resize_image(cropped, min_side=200, max_side=200)
            except:
               cropped = None

            if cropped is not None:
            
                bxs, scs, lbs = small_model.predict_on_batch(np.expand_dims(cropped, axis=0))
                bxs /= scale2
                #cv2.imwrite("./results1/%d.jpg" % x, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                for bs, sc, lb in zip(bxs[0], scs[0], lbs[0]):
                    if sc < 0.8:
                        break
                    #caption = "{} {:.3f}".format(labels_to_names[label], score)
                    #draw_caption(draw, b, caption)
                    #draw_box(draw, b, color=color)
                    bfinal=b
                    break 
        else:
            bfinal = b
        if bfinal is not None:
            csv_file.write("%d,%d,%d,%d,%d,%f\n" % (x, bfinal[0], bfinal[1], bfinal[2], bfinal[3],score))
            draw_box(draw2, b, color=color)
    video.write(draw2)
        
video.release()
