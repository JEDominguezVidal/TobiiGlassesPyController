import cv2
import av
import numpy as np
import csv
from datetime import datetime
import time
from tobiiglassesctrl import TobiiGlassesController
import os
from ultralytics import YOLO

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
        writer.writerow(['Wall time from start [us]', 'Video TS [us]', 'Video PTS [pulses @ 90kHz]', 'Gaze TS [us]', 'Wall time from start delta [us]', 'Video TS delta [us]', 'Video PTS delta [pulses @ 90kHz]', 'Gaze TS delta [us]', 'Gaze to Video TS [us]', 'Wall time from last PTS [us]'])
    return file_path

def update_csv_file(file_path, wall_time_from_start, video_ts, video_pts, gaze_ts, prev_wall_time_from_start, prev_video_ts, prev_video_pts, prev_gaze_ts, wall_time_from_last_pts):
    """
    Updates the .csv file with the new timestamps and their differences.

    Parameters:
    - file_path: str, path of the .csv file.
    - wall_time_from_start: float, wall time from starting this script.
    - video_ts: float, current timestamp of the video.
    - video_pts: float, current pts timestamp of the video.
    - gaze_ts: float, current timestamp of the gaze.
    - prev_wall_time_from_start: float, previous wall time from starting this script.
    - prev_video_ts: float, previous timestamp of the video.
    - prev_video_pts: float, previous pts timestamp of the video.
    - prev_gaze_ts: float, previous gaze timestamp.
    - wall_time_from_last_pts: float, wall time from last PTS update.
    """
    wall_time_from_start_delta = wall_time_from_start - prev_wall_time_from_start if prev_wall_time_from_start is not None else 0
    video_ts_delta = video_ts - prev_video_ts if prev_video_ts is not None else 0
    video_pts_delta = video_pts - prev_video_pts if prev_video_pts is not None else 0
    gaze_ts_delta = gaze_ts - prev_gaze_ts if prev_gaze_ts is not None else 0
    gaze_to_video_ts = gaze_ts - video_ts

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([wall_time_from_start, video_ts, video_pts, gaze_ts, wall_time_from_start_delta, video_ts_delta, video_pts_delta, gaze_ts_delta, gaze_to_video_ts, wall_time_from_last_pts])

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

def apply_gaze_mask_with_bounding_boxes_YOLOv8(frame, gaze_pos, alpha, detections, classes, confidence_threshold=0.5):
    """
    Applies a semi-transparent mask in the cell corresponding to the gaze.

    Parameters:
    - frame: numpy array, video frame.
    - gaze_position: tuple, gaze position (gaze_x, gaze_y).
    - alpha: float, mask transparency (0.0 to 1.0).
    - detections: dict, it contains bounding boxes, class_ids and confidences for each bounding box.
    - classes: list, list of class names.
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

    # Dibujar las detecciones en el frame
    for detection in detections:
        cords = detection.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        if cords[0] < gaze_x < cords[2] and cords[1] < gaze_y < cords[3]:
            confidence = detection.conf[0].item()
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = (cords[0], cords[1], cords[2], cords[3])
                best_class_id = int(detection.cls[0].item())
                best_confidence = confidence

    if best_box and max_confidence >= confidence_threshold:
        x, y, x_end, y_end = best_box
        cv2.rectangle(mask, (x, y), (x_end, y_end), (0, 0, 255), -1)
        label = f'{classes[best_class_id]} ({best_confidence:.2f})'
        draw_label(frame, label, (x, y - 10))

    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, frame)
    return frame

def apply_bounding_boxes_YOLOv8(frame, gaze_pos, alpha, detections, classes, confidence_threshold=0.5):
    """
    Draw all the bounding boxes detected by YOLOv8 in red and the one corresponding to the gaze in green.

    Parameters:
    - frame: numpy array, video frame.
    - gaze_position: tuple, gaze position (gaze_x, gaze_y).
    - alpha: float, bounding boxes transparency (0.0 to 1.0).
    - detections: dict, it contains bounding boxes, class_ids and confidences for each bounding box.
    - classes: list, list of class names.
    - confidence_threshold: float, minimum confidence to accept a bounding box.

    Return:
    - frame: numpy array, frame with bounding boxes applied.
    """
    gaze_x, gaze_y = gaze_pos
    mask = frame.copy()
    max_confidence = 0
    best_box = None
    best_class_id = None
    best_confidence = None
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    # Draw all detections in red and find the best detection for the gaze
    for detection in detections:
        cords = detection.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        confidence = detection.conf[0].item()
        class_id = int(detection.cls[0].item())
        label = f'{classes[class_id]} ({confidence:.2f})'
        
        # Draw all the bounding boxes in red
        cv2.rectangle(mask, (cords[0], cords[1]), (cords[2], cords[3]), red_color, 2)
        draw_label(frame, label, (cords[0], cords[1] - 10), font_color=red_color)
        
        # Check if gaze is within the bounding box
        if cords[0] < gaze_x < cords[2] and cords[1] < gaze_y < cords[3] and confidence > max_confidence:
            max_confidence = confidence
            best_box = (cords[0], cords[1], cords[2], cords[3])
            best_class_id = class_id
            best_confidence = confidence

    # If there is a bounding box containing the gaze with greater confidence, paint it in green
    if best_box and max_confidence >= confidence_threshold:
        x, y, x_end, y_end = best_box
        cv2.rectangle(mask, (x, y), (x_end, y_end), green_color, 4)
        label = f'{classes[best_class_id]} ({best_confidence:.2f})'
        draw_label(frame, label, (x, y - 10), font_color=green_color)

    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, frame)
    return frame


# Main function
def main():
    # Initial configuration
    ipv4_address = "192.168.100.10"
    model_path = 'fine_tuned_models/best_small_epoch_16.pt'
    model = YOLO(model_path)
    classes = ['phone', 'scissors', 'apple']

    alpha = 0.4  # Mask transparency
    confidence_threshold = 0.6  # Minimum confidence for bounding boxes

    prev_time_from_start = None
    prev_video_ts = None
    prev_video_pts = None
    prev_gaze_ts = None
    wall_time_from_last_pts = 0.0 
    wall_time_from_last_pts_acc = 0.0

    start_current_time = time.time()
    tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)
    tobiiglasses.start_streaming()

    csv_file_path = create_csv_file()

    rtsp_url = f"rtsp://{ipv4_address}:8554/live/scene"
    container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
    stream = container.streams.video[0]

    # Read until video is completed
    for frame in container.decode(stream):
        data_gp = tobiiglasses.get_data()['gp']
        data_pts = tobiiglasses.get_data()['pts']
        frame_cv = frame.to_ndarray(format='bgr24')

        if data_gp['ts'] > 0 and data_pts['ts'] > 0:
            current_time = time.time()
            time_from_start = round((current_time - start_current_time) * 1000000)  # time from beginning in microseconds
            height, width = frame_cv.shape[:2]

            current_gaze_timestamp = data_gp['ts']
            gaze_x, gaze_y = int(data_gp['gp'][0] * width), int(data_gp['gp'][1] * height)

            # Generate circle for gaze
            cv2.circle(frame_cv, (gaze_x, gaze_y), 20, (0, 0, 255), 6)

            # Detect objets using fine-tuned YOLOv8
            results = model(frame_cv)
            detections = results[0].boxes
            #frame_cv = apply_gaze_mask_with_bounding_boxes_YOLOv8(frame_cv, (gaze_x, gaze_y), alpha, detections, classes, confidence_threshold)
            frame_cv = apply_bounding_boxes_YOLOv8(frame_cv, (gaze_x, gaze_y), alpha, detections, classes, confidence_threshold)

            # Update .csv file
            current_video_timestamp_ts = data_pts['ts']
            current_video_timestamp_pts = data_pts['pts']
            video_pts_delta = current_video_timestamp_pts - prev_video_pts if prev_video_pts is not None else 0
            if video_pts_delta > 0:
                wall_time_from_last_pts = wall_time_from_last_pts_acc + (time_from_start - prev_time_from_start) if prev_time_from_start is not None else 0
                wall_time_from_last_pts_acc = 0.0
            else:
                wall_time_from_last_pts = 0.0
                wall_time_from_last_pts_acc = wall_time_from_last_pts_acc + (time_from_start - prev_time_from_start) if prev_time_from_start is not None else 0

            update_csv_file(csv_file_path, time_from_start, current_video_timestamp_ts, current_video_timestamp_pts, current_gaze_timestamp, prev_time_from_start, prev_video_ts, prev_video_pts, prev_gaze_ts, wall_time_from_last_pts)

            # Update previous timestamps
            prev_time_from_start = time_from_start
            prev_video_ts = current_video_timestamp_ts
            prev_video_pts = current_video_timestamp_pts
            prev_gaze_ts = current_gaze_timestamp

        # Show frame
        cv2.imshow('Video', frame_cv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    container.close()
    cv2.destroyAllWindows()
    tobiiglasses.stop_streaming()
    tobiiglasses.close()

if __name__ == "__main__":
    main()
