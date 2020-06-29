from keras.models import Model
from keras.layers import Concatenate
import sys
sys.path.insert(0, '../')

import keras 
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.seq_nms import seq_nms, BoxGraph
from keras_retinanet.utils.nms import non_max_suppression_fast
import keras_retinanet.backend as backend
import keras_retinanet.layers as layers

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import pickle 
from argparse import ArgumentParser

import multiprocessing

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

num_classes = 80
def post_process_nms_applied(bg, labels, max_detections=300, no_seq_nms=False, score_metric='avg'):
    print("boxes", bg.boxes.shape)
    print("scores", bg.scores.shape)
    bg.seq_nms()
    #print(classifications[0][0])
    indices = []
    for i in range(bg.boxes.shape[0]):
        frame_indices = non_max_suppression_fast(bg.boxes[i,:,:], bg.scores[i,:], overlapThresh=0.5)
        frame_indices = [[i, idx] for idx in frame_indices]
        indices.append(frame_indices)
    indices = keras.backend.concatenate(indices, axis=0)

    scores              = backend.gather_nd(bg.scores, indices)
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))
    
    indices             = keras.backend.gather(indices, top_indices)
    boxes               = backend.gather_nd(bg.boxes, indices)
    labels              = backend.gather_nd(labels, indices)


    indices=indices.numpy()  
    boxes=boxes.numpy()
    scores=scores.numpy()
    labels=labels.numpy()
    return boxes, scores, labels, indices

def post_process(boxes, classifications, max_detections=300):
    all_indices = []
    #perform per class filtering
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')

        seq_nms(boxes, scores)
        class_indices = non_max_suppression_fast(boxes, scores, overlapThresh=0.5)

        all_indices.append(class_indices)

    # concatenate indices to single tensor
    indices = keras.backend.concatenate(all_indices, axis=0)
    bg = BoxGraph()
    for f in range(boxes.shape[0]):
        bg.add_layer(np.expand_dims(boxes[f,:,:], axis=0), np.expand_dims(scores[f,:,:]))
    
    bg.seq_nms()
            
    
    # select top k
    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return boxes, scores, labels 


def capture_video_frames(video_file, num_frames_to_capture=3, start_fpos=0):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fpos) #570
    captured = 0
    frames = []
    draw_frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180) 
        #cv2.imshow('frame',gray)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        # collect drawing frames 
        draw = frame.copy()
        draw_frames.append(draw)

        # preprocess frames 
        image = preprocess_image(frame)
        image, scale = resize_image(image)
        #print(image.shape)

        frames.append(image)
        captured += 1 
        if captured == num_frames_to_capture: break

    cap.release()
    cv2.destroyAllWindows()

    all_frames = np.stack(frames)
    draw_frames = np.stack(draw_frames)
    print("Image batch:", all_frames.shape)
    return all_frames, draw_frames, scale

def build_model(model_file, add_filtering=True, max_detections=100, nms_threshold=0.6, score_threshold=0.05, training_model=False, filtering_layer_included=False):
    model_path = os.path.join('..', 'snapshots', model_file)
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    if training_model:
        model = models.convert_model(model)
    #print(model.summary())

    if filtering_layer_included:
        print("model already contains filtered layer")
        boxes, classification = model.layers[-3].output, model.layers[-2].output
    else:
        boxes, classification = model.layers[-2].output, model.layers[-1].output

    if add_filtering:
        #boxes, classification = model.layers[-3].output, model.layers[-2].output
        #other = model.layers[-2].output
        print("Adding modified filtering layer...")
        detections = layers.FilterDetections(
                nms                   = True,
                class_specific_filter = True,
                name                  = 'filtered_detections',
                nms_threshold         = nms_threshold,
                score_threshold       = score_threshold,
                max_detections        = max_detections,
                parallel_iterations   = 32
            ) ([boxes, classification])
       

        # define model with newly defined filtering layer
        model1 = Model(model.input, detections)
    else:
        # get classification and regression layers and build new model 
        model1 = Model(model.input, outputs=[boxes, classification])

    #print(model1.summary())

    return model1

def visualize_video(box, score, label, index, scale, pause_rate=3):
    box /= scale
    frame_idx = index[0]
    #print(box, score, label)
    draw = draw_frames[frame_idx]
    if score < 0.5: return

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)
            
    caption = "{} {:.3f}".format(labels_to_names.get(label, "misc"), score)
    draw_caption(draw, b, caption)
    #video.write(draw)

def parse_args():
    """ Parse the arguments.
    """
    parser = ArgumentParser(description='Simple script for running retinanet on a video')
    parser.add_argument('--model_file', help='File containing saved retinaNet model weights', required=True)
    parser.add_argument('--video_file', help='Video file to run the model on', required=True)
    parser.add_argument('--frames_to_capture', help='Number of sequential video frames to capture and detect objects on', default=3, type=int)
    parser.add_argument('--start_fpos', help='Starting frame at which to begin capturing frames', default=0, type=int)
    parser.add_argument('--add_filtering', help='Flag indicating whether to include the final filtering layer in the loaded model', action='store_true')
    parser.add_argument('--pause_length', help='Number of seconds to pause betweeing displaying frames for visualized detections', default=1, type=int)
    parser.add_argument('--linkage_threshold', help='Threshold to use for box linkage threshold in seq-nms algorithm', default=0.5, type=float)
    parser.add_argument('--score_threshold', help='Threshold to use as minimum confidence for score confidences', default=0.05, type=float)
    parser.add_argument('--nms_threshold', help='Threshold to use as for nms algorithm', default=0.6, type=float)
    parser.add_argument('--seq_nms_threshold', help='Threshold to use for suppression of boxes with regards to best sequences in seq-nms algorithm', default=0.4, type=float)
    parser.add_argument('--max_detections', help='Maximum number of detections to return from post-processing', default=500, type=int)
    parser.add_argument('--training_model', help='Flag indicating whether model is training model or not', action='store_true')
    parser.add_argument('--filter_layer_included', help='Flag indicating whether loaded model already contains a filtering level', action='store_true')
    parser.add_argument('--general_detection', help='Flag indicating whether loaded model is general fine-tuned for ball dectection', action='store_true')
    parser.add_argument('--no_seq_nms', help='Flag to turn off seq-nms', action='store_true')
    parser.add_argument('--score_metric', help='Scoring metric to use for rescoring best sequence in seq-nms algorithm', default='avg', type=str)
    args = parser.parse_args()
    return args 

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.box_graph = BoxGraph()

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            print('%s: %s' % (proc_name, next_task))
            boxes, scores = next_task()
            start = time.time()
            self.box_graph.add_layer(boxes, scores)
            print("add layer", time.time() - start)
            self.task_queue.task_done()
            #self.result_queue.put(answer)
        self.result_queue.put(self.box_graph)

class Task(object):
    def __init__(self, boxes, scores):
        self.boxes = boxes
        self.scores = scores
    def __call__(self):
        return self.boxes, self.scores
    def __str__(self):
        return 'Task'


if __name__ == '__main__':
    # use this to change which GPU to use
    args = parse_args()
    gpu = 0

    # set the modified tf session as backend in keras
    setup_gpu(gpu)

    if args.general_detection:
        labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    else:
        labels_to_names = {0:'ball'}


    task_queue, result_queue = multiprocessing.JoinableQueue(), multiprocessing.Queue()

    consumer = Consumer(task_queue, result_queue)
    consumer.start()

    # get video frames, draw frames and scale

    image_batch, draw_frames, scale = capture_video_frames(args.video_file, num_frames_to_capture=args.frames_to_capture, start_fpos=args.start_fpos)

    #build model 
    model = build_model(args.model_file, add_filtering=args.add_filtering, max_detections=args.max_detections, nms_threshold=args.nms_threshold, 
                score_threshold=args.score_threshold, training_model=args.training_model, filtering_layer_included=args.filter_layer_included)

    # process frames 
    model.predict_on_batch(np.expand_dims(image_batch[0], axis=0))
    start = time.time()
    labels = np.array([])
    for x in range(0, image_batch.shape[0]-1, 2):
        print("step", x)
        start1 = time.time()
        b, c, l = model.predict_on_batch(image_batch[x:x+2])
        print("inf time:", time.time() - start1)
        for i in range(2):
            task_queue.put(Task(np.expand_dims(b[i], axis=0), np.expand_dims(c[i], axis=0)))
        if labels.size == 0:
            labels = l
        else:
            labels = np.concatenate((labels, l))
    task_queue.put(None)
    task_queue.join()
    print("processing time1: ", time.time() - start)
    start = time.time()
  
    #boxes, classifications, labels = model.predict_on_batch(image_batch)

    graph = result_queue.get()


    #np.save("boxes15.npy", boxes)
    #np.save("classifcations15.npy", classification)

    # Post-processing
    print(graph.boxes)
    start = time.time()

    if not args.add_filtering:
        boxes, scores, labels, inds = post_process(boxes, classification)
    else:
        boxes, scores, labels, inds = post_process_nms_applied(graph, labels, no_seq_nms=args.no_seq_nms, score_metric=args.score_metric)
    #print(boxes.shape, scores.shape, labels.shape)
    print("post-processing time: ", time.time() - start)
    # visualize detections
    #print("inds", inds)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("out.mp4", fourcc,30, (1920, 1080));
    for idx in range(boxes.shape[0]):
        visualize_video(boxes[idx,:], scores[idx],
                labels[idx], inds[idx], scale, pause_rate=args.pause_length)

    for frame in draw_frames:
        video.write(frame)
    video.release()
    '''
    for frame_idx, draw in enumerate(draw_frames):
        boxes_f, scores_f, labels_f = boxes[frame_idx], scores[frame_idx], labels[frame_idx]
        for box, score, label in zip(boxes_f, scores_f, labels_f):
            # scores are sorted so we can break
            if score < 0.5:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    '''

