# live_scene_gaze_and_aoi.py : A demo for video streaming, gaze and synchronized Area of Interest (AoI)
#
# Copyright (C) 2018  Davide De Tommaso
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import cv2
import numpy as np
import csv
from datetime import datetime
import time
from tobiiglassesctrl import TobiiGlassesController
import os



def load_yolo_model_gpu(cfg_path, weights_path, names_path, use_cuda=True):
    """
    Loads the YOLO model in GPU from configuration files, weights and class names.

    Parameters:
    - cfg_path: str, path to the YOLO configuration file (.cfg).
    - weights_path: str, path to YOLO weights file (.weights).
    - names_path: str, path to the YOLO class name file (.names).
    - use_cuda: bool, select if the model should use GPU or not.

    Return:
    - net: cv2.dnn_Net, neural network loaded.
    - classes: list, list of class names.
    - output_layers: list, list of network output layer names.
    """
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers



def load_yolo_model(cfg_path, weights_path, names_path):
    """
    Loads the YOLO model from configuration files, weights and class names.

    Parameters:
    - cfg_path: str, path to the YOLO configuration file (.cfg).
    - weights_path: str, path to YOLO weights file (.weights).
    - names_path: str, path to the YOLO class name file (.names).

    Return:
    - net: cv2.dnn_Net, neural network loaded.
    - classes: list, list of class names.
    - output_layers: list, list of network output layer names.
    """
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers



def detect_objects_yolo(frame, confidence_threshold=0.5):
    """
    Detects objects in the frame using YOLO and returns the bounding boxes and class IDs.

    Parameters:
    - frame: numpy array, video frame.
    - confidence_threshold: float, minimum confidence to accept a detected object.

    Return:
    - bounding_boxes: list, array with the bounding boxes of the detected objects.
    - final_class_ids: list, array with the class IDs of the detected objects.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    bounding_boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bounding_boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confidence_threshold, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    final_boxes = [bounding_boxes[i] for i in indexes]
    final_class_ids = [class_ids[i] for i in indexes]

    return final_boxes, final_class_ids



def detect_objects_yolo(frame, confidence_threshold=0.5):
    """
    Detects objects in the frame using YOLO and returns the bounding boxes, class IDs, and confidences.

    Parameters:
    - frame: numpy array, video frame.
    - confidence_threshold: float, minimum confidence to accept a detected object.

    Return:
    - bounding_boxes: list, array with the bounding boxes of the detected objects.
    - final_class_ids: list, array with the class IDs of the detected objects.
    - final_confidences: list, array with the confidences of the detected objects.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    bounding_boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bounding_boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confidence_threshold, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    final_boxes = [bounding_boxes[i] for i in indexes]
    final_class_ids = [class_ids[i] for i in indexes]
    final_confidences = [confidences[i] for i in indexes]

    return final_boxes, final_class_ids, final_confidences



def draw_label(frame, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(0, 255, 0), font_thickness=2):
    """
    Draws a label on the video frame.

    Parameters:
    - frame: numpy array, video frame.
    - text: str, text to display.
    - pos: tuple, position (x, y) for the label.
    - font: cv2.FONT, font type (optional).
    - font_scale: float, font size (optional).
    - font_color: tuple, font color (optional).
    - font_thickness: int, font thickness (optional).
    """
    cv2.putText(frame, text, pos, font, font_scale, font_color, font_thickness)



def generate_bounding_boxes(frame, cell_size=(3, 3)):
    """
    Generates an array of bounding boxes by dividing the frame into an array of cells.

    Parameters:
    - frame: numpy array, video frame.
    - cell_size: tuple, size of the cell array (default: (3, 3)).

    Return:
    - bounding_boxes: list, array with the bounding boxes for each cell.
    """
    height, width, _ = frame.shape
    num_rows, num_cols = cell_size

    cell_width = width // num_cols
    cell_height = height // num_rows

    bounding_boxes = []
    for row in range(num_rows):
        for col in range(num_cols):
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            bounding_boxes.append((x_start, y_start, x_end, y_end))

    return bounding_boxes



def apply_gaze_mask_with_bounding_boxes(frame, gaze_pos, alpha, bounding_boxes, class_ids, classes, confidences, confidence_threshold=0.5):
    """
    Applies a semi-transparent mask in the cell corresponding to the gaze.

    Parameters:
    - frame: numpy array, video frame.
    - gaze_position: tuple, gaze position (gaze_x, gaze_y).
    - alpha: float, mask transparency (0.0 to 1.0).
    - bounding_boxes: list, array with the bounding boxes of detected objects.
    - class_ids: list, array with the class IDs of the detected objects.
    - classes: list, list of class names.
    - confidences: list, list of confidences for each detected object.
    - confidence_threshold: float, minimum confidence to accept a bounding box.

    Return:
    - frame: numpy array, frame with mask applied.
    """
    gaze_x, gaze_y = gaze_pos
    mask = frame.copy()
    max_confidence = 0
    best_box = None
    best_class_id = None
    best_confidence = None

    for i, (x, y, x_end, y_end) in enumerate(bounding_boxes):
        if x < gaze_x < x_end and y < gaze_y < y_end:
            confidence = confidences[i]
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = (x, y, x_end, y_end)
                best_class_id = class_ids[i]
                best_confidence = confidence

    if best_box and max_confidence >= confidence_threshold:
        x, y, x_end, y_end = best_box
        cv2.rectangle(mask, (x, y), (x_end, y_end), (0, 0, 255), -1)
        # Draw the class label with confidence
        label = f"{classes[best_class_id]} ({best_confidence:.2f})"
        draw_label(frame, label, (x, y - 10))

    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, frame)
    return frame



def apply_gaze_mask(frame, gaze_position, alpha, matrix_size=(3, 3)):
    """
    Applies a semi-transparent mask to the Area of Interest (AoI) corresponding to the gaze in a AoIs matrix.

    Parameters:
    - frame: numpy array, video frame.
    - gaze_position: tuple, gaze position (gaze_x, gaze_y).
    - alpha: float, mask transparency (0.0 to 1.0).
    - matrix_size: tuple, AoI matrix size (default: (3, 3)).

    Return:
    - frame: numpy array, frame with mask applied.
    """
    height, width, _ = frame.shape
    gaze_x, gaze_y = gaze_position
    num_rows, num_cols = matrix_size

    # Calculate the size of each AoI
    cell_width = width // num_cols
    cell_height = height // num_rows

    # Calculate in which AoI is the gaze located
    col = gaze_x // cell_width
    row = gaze_y // cell_height

    # Calculate AoI coordinates
    x_start = col * cell_width
    y_start = row * cell_height
    x_end = x_start + cell_width
    y_end = y_start + cell_height

    # Create semi-transparent mask
    mask = frame.copy()
    mask[y_start:y_end, x_start:x_end] = (0, 0, 255)  # Rojo

    # Aplicar la máscara con transparencia
    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, frame)

    return frame



def create_csv_file():
    """
    Creates a .csv file with a unique name based on the current date and time.

    Return:
    - file_path: str, path to the created .csv file.
    """
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = f'logs/gaze_data_{now}.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Wall time from start [us]', 'Video TS (cv2) [ms]', 'Video TS [us]', 'Video PTS [pulses @ 90kHz]', 'Gaze TS [us]', 'Wall time from start delta [us]', 'Video TS (cv2) delta [ms]', 'Video TS delta [us]', 'Video PTS delta [pulses @ 90kHz]', 'Gaze TS delta [us]', 'Gaze to Video TS [us]', 'Wall time from last PTS [us]'])
    return file_path



def update_csv_file(file_path, wall_time_from_start, video_ts_cv2, video_ts, video_pts, gaze_ts, prev_wall_time_from_start, prev_video_ts_cv2, prev_video_ts, prev_video_pts, prev_gaze_ts, wall_time_from_last_pts):
    """
    Updates the .csv file with the new timestamps and their differences.

    Parameters:
    - file_path: str, path of the .csv file.
    - wall_time_from_start: float, wall time from starting this script.
    - video_ts_cv2: float, current timestamp of the video obtained from OpenCV2.
    - video_ts: float, current timestamp of the video.
    - video_pts: float, current pts timestamp of the video.
    - gaze_ts: float, current timestamp of the gaze.
    - prev_wall_time_from_start: float, previous wall time from starting this script.
    - prev_video_ts_cv2: float, previous timestamp of the video obtained from OpenCV2.
    - prev_video_ts: float, previous timestamp of the video.
    - prev_video_pts: float, previous pts timestamp of the video.
    - prev_gaze_ts: float, previous gaze timestamp.
    - wall_time_from_last_pts: float, wall time from last PTS update.
    """
    wall_time_from_start_delta = wall_time_from_start - prev_wall_time_from_start if prev_wall_time_from_start is not None else 0
    video_ts_cv2_delta = video_ts_cv2 - prev_video_ts_cv2 if prev_video_ts_cv2 is not None else 0
    video_ts_delta = video_ts - prev_video_ts if prev_video_ts is not None else 0
    video_pts_delta = video_pts - prev_video_pts if prev_video_pts is not None else 0
    gaze_ts_delta = gaze_ts - prev_gaze_ts if prev_gaze_ts is not None else 0
    gaze_to_video_ts = gaze_ts - video_ts



    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([wall_time_from_start, video_ts_cv2, video_ts, video_pts, gaze_ts, wall_time_from_start_delta, video_ts_cv2_delta, video_ts_delta, video_pts_delta, gaze_ts_delta, gaze_to_video_ts, wall_time_from_last_pts])








#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


# Initial configuration
ipv4_address = "192.168.100.10"

cfg_path = "yolo/yolov3.cfg"
weights_path = "yolo/yolov3.weights"
cfg_tiny_path = "yolo/yolov3-tiny.cfg"
weights_tiny_path = "yolo/yolov3-tiny.weights"
names_path = "yolo/coco.names"
use_tiny_yolo = True

alpha = 0.5  # Mask transparency
confidence_threshold = 0.25  # Minimum confidence for bounding boxes

prev_time_from_start = None
prev_video_ts_cv2 = None
prev_video_ts = None
prev_video_pts = None
prev_gaze_ts = None
wall_time_from_last_pts = 0.0 
wall_time_from_last_pts_acc = 0.0 


# Main code beginning

start_current_time = time.time()
tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)

if use_tiny_yolo:
    net, classes, output_layers = load_yolo_model(cfg_tiny_path, weights_tiny_path, names_path)
else:
    net, classes, output_layers = load_yolo_model(cfg_path, weights_path, names_path)

cap = cv2.VideoCapture("rtsp://%s:8554/live/scene" % ipv4_address)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

csv_file_path = create_csv_file()

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

tobiiglasses.start_streaming()
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    print("-------------------------")
    current_time = time.time()
    time_from_start = round((current_time - start_current_time) * 1000000) # time from beginning in microseconds
    print("Current time: ", current_time)
    print("Time from start: ", time_from_start)
    current_video_timestamp_cv2 = cap.get(cv2.CAP_PROP_POS_MSEC)
    print("Current video timestamp (from cv2): ", current_video_timestamp_cv2)
    timestamps.append(current_video_timestamp_cv2)
    height, width = frame.shape[:2]

    data_gp  = tobiiglasses.get_data()['gp']
    data_pts  = tobiiglasses.get_data()['pts']
    print(data_pts)

    if data_pts['ts'] > 0:
        # Iterate over dictionary keys
        for key, value in data_pts.items():
            print(f'{key}: {value}')

        print(data_pts['pts'])
        
    print(data_gp)

    if data_gp['ts'] > 0:
        current_gaze_timestamp = data_gp['ts']
        print("Current gaze timestamp:", current_gaze_timestamp)
        gaze_x, gaze_y = int(data_gp['gp'][0] * width), int(data_gp['gp'][1] * height)
        
        # Generate circle for gaze
        cv2.circle(frame,(gaze_x,gaze_y), 60, (0,0,255), 6)

        # # Generate AoI mask
        # frame = apply_gaze_mask(frame, (gaze_x, gaze_y), alpha)

        # # Generate bounding boxes
        # bounding_boxes = generate_bounding_boxes(frame)

        # Detect objects using YOLO
        bounding_boxes, class_ids, confidences = detect_objects_yolo(frame, confidence_threshold)

        # Apply mask with bounding boxes
        frame = apply_gaze_mask_with_bounding_boxes(frame, (gaze_x, gaze_y), alpha, bounding_boxes, class_ids, classes, confidences, confidence_threshold)

        # Update .csv file
        current_video_timestamp_ts = data_pts['ts']
        current_video_timestamp_pts = data_pts['pts']
        video_pts_delta = current_video_timestamp_pts - prev_video_pts if prev_video_pts is not None else 0
        if video_pts_delta > 0:
            wall_time_from_last_pts = wall_time_from_last_pts_acc + (time_from_start-prev_time_from_start) if prev_time_from_start is not None else 0
            wall_time_from_last_pts_acc = 0.0
        else:
            wall_time_from_last_pts = 0.0
            wall_time_from_last_pts_acc = wall_time_from_last_pts_acc + (time_from_start-prev_time_from_start) if prev_time_from_start is not None else 0

        update_csv_file(csv_file_path, time_from_start, current_video_timestamp_cv2, current_video_timestamp_ts, current_video_timestamp_pts, current_gaze_timestamp, prev_time_from_start, prev_video_ts_cv2, prev_video_ts, prev_video_pts, prev_gaze_ts, wall_time_from_last_pts)

        # Update previous timestamps
        prev_time_from_start = time_from_start
        prev_video_ts_cv2 = current_video_timestamp_cv2
        prev_video_ts = current_video_timestamp_ts
        prev_video_pts = current_video_timestamp_pts
        prev_gaze_ts = current_gaze_timestamp



    # Display the resulting frame
    cv2.imshow('Tobii Pro Glasses 2 - Live Scene',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
tobiiglasses.stop_streaming()
tobiiglasses.close()
