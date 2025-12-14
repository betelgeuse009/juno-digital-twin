import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import torchvision # For NMS

# --- Configuration ---
# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Stop Utils - Using device: {DEVICE}")

# --- Control for internal drawing (similar to original 'SHOW' behavior) ---
# If True, 'analysis' will draw on a copy of the frame and return it.
# If False, 'analysis' will return the original frame.
# This can be set externally if this module is imported.
SHOW_INTERNAL_VISUALS = True # Set this to False if the caller handles all drawing

# --- Stage 1: Detector Model ---
DETECTOR_MODEL_PATH = "/home/bylogix/Downloads/stop_weights.pt"
DETECTOR_CONF_THRESHOLD = 0.20
DETECTOR_STOP_SIGN_CLASS_ID = 0
INFERENCE_SIZE_DETECTOR = (640, 640)

# --- Stage 2: Verifier Model ---
VERIFIER_MODEL_PATH = "/home/bylogix/Downloads/yolo12x.pt"
VERIFIER_CONF_THRESHOLD = 0.50
VERIFIER_STOP_SIGN_CLASS_NAME = "stop sign"

# --- Slicing Configuration ---
SLICE_ROWS = 2
SLICE_COLS = 4
OVERLAP_RATIO = 0.1

# --- NMS ---
NMS_IOU_THRESHOLD = 0.45

# --- Visual Alarm Parameters (used if SHOW_INTERNAL_VISUALS is True) ---
ALARM_COLOR = (0, 0, 255)
ALARM_THICKNESS = 10
ALARM_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALARM_FONT_SCALE = 1.0
ALARM_TEXT = "STOP SIGN!"

# --- Load Models ---
detector_model = YOLO(DETECTOR_MODEL_PATH).to(DEVICE)
verifier_model = None
verifier_stop_sign_id = None

try:
    print(f"Loading detector model from: {DETECTOR_MODEL_PATH}")
    print(f"Detector model '{DETECTOR_MODEL_PATH}' loaded successfully.")
    if not hasattr(detector_model, 'names') or not detector_model.names:
        detector_model.names = {DETECTOR_STOP_SIGN_CLASS_ID: VERIFIER_STOP_SIGN_CLASS_NAME}
    elif DETECTOR_STOP_SIGN_CLASS_ID not in detector_model.names:
        print(f"Warning: Detector class ID {DETECTOR_STOP_SIGN_CLASS_ID} not in names.")

    print(f"Loading verifier model from: {VERIFIER_MODEL_PATH}")
    verifier_model = YOLO(VERIFIER_MODEL_PATH).to(DEVICE)
    print(f"Verifier model '{VERIFIER_MODEL_PATH}' loaded successfully.")
    try:
        verifier_stop_sign_id = [k for k, v in verifier_model.names.items() if v.lower() == VERIFIER_STOP_SIGN_CLASS_NAME.lower()][0]
        print(f"Verifier '{VERIFIER_STOP_SIGN_CLASS_NAME}' class ID: {verifier_stop_sign_id}")
    except IndexError:
        print(f"ERROR: Class '{VERIFIER_STOP_SIGN_CLASS_NAME}' not in verifier names: {verifier_model.names}")
        verifier_model = None
        print("Verifier model disabled.")
    except Exception as e:
        print(f"Error loading verifier model names: {e}")
        verifier_model = None
        print("Verifier model disabled.")
except Exception as e:
    print(f"Error loading models: {e}")


def slice_frame_new(frame, rows, cols, overlap_ratio):
    # (Same slice_frame_new function as before - kept for brevity in this explanation)
    height, width = frame.shape[:2]
    if height == 0 or width == 0: return []
    slice_height_base = height // rows
    slice_width_base = width // cols
    overlap_h = int(slice_height_base * overlap_ratio)
    overlap_w = int(slice_width_base * overlap_ratio)
    slices_meta = []
    for i in range(rows):
        for j in range(cols):
            y_start_slice, y_end_slice = i * slice_height_base, (i + 1) * slice_height_base
            x_start_slice, x_end_slice = j * slice_width_base, (j + 1) * slice_width_base
            y_start_overlap, y_end_overlap = max(0, y_start_slice - overlap_h // 2), min(height, y_end_slice + overlap_h // 2)
            x_start_overlap, x_end_overlap = max(0, x_start_slice - overlap_w // 2), min(width, x_end_slice + overlap_w // 2)
            if i == 0: y_start_overlap = 0
            if i == rows - 1: y_end_overlap = height
            if j == 0: x_start_overlap = 0
            if j == cols - 1: x_end_overlap = width
            sliced_img = frame[y_start_overlap:y_end_overlap, x_start_overlap:x_end_overlap]
            if sliced_img.size == 0 or sliced_img.shape[0] < 10 or sliced_img.shape[1] < 10: continue
            slices_meta.append({
                'image': sliced_img,
                'coords_on_original': (x_start_overlap, y_start_overlap, x_end_overlap, y_end_overlap)
            })
    return slices_meta

def analysis(frame_original):
    """
    Performs two-stage stop sign detection.
    Returns: (bool: stop_sign_found, frame_output)
             frame_output is frame_original if SHOW_INTERNAL_VISUALS is False or no sign found.
             frame_output is a copy of frame_original with drawings if SHOW_INTERNAL_VISUALS is True and sign found.
    """
    if detector_model is None or verifier_model is None or verifier_stop_sign_id is None:
        return False, frame_original

    output_frame = frame_original # Default to returning the original frame
    original_height, original_width = frame_original.shape[:2]

    slices_meta = slice_frame_new(frame_original, SLICE_ROWS, SLICE_COLS, OVERLAP_RATIO)
    batch_slice_images, valid_slices_meta = [], []
    for s_meta in slices_meta:
        if s_meta['image'] is not None and s_meta['image'].shape[0] > 0 and s_meta['image'].shape[1] > 0:
            batch_slice_images.append(s_meta['image'])
            valid_slices_meta.append(s_meta)
    if not batch_slice_images: return False, frame_original

    # --- Stage 1: Detector ---
    detector_results = detector_model.predict(batch_slice_images, imgsz=INFERENCE_SIZE_DETECTOR,
                                              conf=DETECTOR_CONF_THRESHOLD, classes=[DETECTOR_STOP_SIGN_CLASS_ID],
                                              device=DEVICE, verbose=False)
    all_boxes_coords_nms, all_scores_nms, all_info_nms = [], [], []
    for i, res_slice in enumerate(detector_results):
        meta = valid_slices_meta[i]
        sx, sy, _, _ = meta['coords_on_original']
        for box in res_slice.boxes:
            x1s,y1s,x2s,y2s = box.xyxy[0].cpu().numpy().astype(int)
            x1f,y1f,x2f,y2f = max(0,x1s+sx),max(0,y1s+sy),min(original_width,x2s+sx),min(original_height,y2s+sy)
            if x2f > x1f and y2f > y1f:
                all_boxes_coords_nms.append([x1f,y1f,x2f,y2f])
                all_scores_nms.append(box.conf.item())
                all_info_nms.append({'box_on_full_frame':[x1f,y1f,x2f,y2f],'detector_conf':box.conf.item()})
    
    potential_detections = []
    if all_boxes_coords_nms:
        boxes_t = torch.tensor(all_boxes_coords_nms,dtype=torch.float32,device=DEVICE)
        scores_t = torch.tensor(all_scores_nms,dtype=torch.float32,device=DEVICE)
        indices = torchvision.ops.nms(boxes_t, scores_t, NMS_IOU_THRESHOLD).cpu().numpy()
        for idx in indices: potential_detections.append(all_info_nms[idx])
    
    # --- Stage 2: Verifier ---
    confirmed_stop_signs_details = []
    rois, rois_orig_info = [], []
    if potential_detections:
        for det_info in potential_detections:
            x1,y1,x2,y2 = [int(c) for c in det_info['box_on_full_frame']]
            roi = frame_original[y1:y2,x1:x2]
            if roi.size > 0 and roi.shape[0]>10 and roi.shape[1]>10:
                rois.append(roi); rois_orig_info.append(det_info)
    
    if rois:
        verifier_results = verifier_model.predict(rois, conf=VERIFIER_CONF_THRESHOLD,
                                                  classes=[verifier_stop_sign_id], device=DEVICE, verbose=False)
        for i, res_roi in enumerate(verifier_results):
            orig_info = rois_orig_info[i]
            max_v_conf = 0.0
            found_in_roi = False
            for box_v in res_roi.boxes: # Check if any detection passed verifier's own threshold
                found_in_roi = True
                max_v_conf = max(max_v_conf, box_v.conf.item())
                break
            if found_in_roi:
                confirmed_stop_signs_details.append({
                    'box': orig_info['box_on_full_frame'],
                    'detector_conf': orig_info['detector_conf'],
                    'verifier_conf': max_v_conf
                })

    stop_sign_found_this_frame = len(confirmed_stop_signs_details) > 0

    if stop_sign_found_this_frame and SHOW_INTERNAL_VISUALS:
        output_frame = frame_original.copy() # Draw on a copy
        for det in confirmed_stop_signs_details:
            x1, y1, x2, y2 = [int(c) for c in det['box']]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), ALARM_COLOR, 2)
            label = f"D:{det['detector_conf']:.2f} V:{det['verifier_conf']:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALARM_COLOR, 2)
        
        h_alarm, w_alarm = output_frame.shape[:2]
        cv2.rectangle(output_frame, (0,0), (w_alarm-1, h_alarm-1), ALARM_COLOR, ALARM_THICKNESS)
        text_size_alarm = cv2.getTextSize(ALARM_TEXT, ALARM_FONT, ALARM_FONT_SCALE, 2)[0]
        text_x = (w_alarm - text_size_alarm[0]) // 2
        text_y = ALARM_THICKNESS + text_size_alarm[1] + 10
        cv2.putText(output_frame, ALARM_TEXT, (text_x, text_y),
                   ALARM_FONT, ALARM_FONT_SCALE, (255,255,255), 2, cv2.LINE_AA)

    return stop_sign_found_this_frame, output_frame


# --- Example Usage (for testing stop_utils.py directly) ---
if True:
    # For testing, you can toggle SHOW_INTERNAL_VISUALS here
    # global SHOW_INTERNAL_VISUALS # If you need to modify it from here
    # SHOW_INTERNAL_VISUALS = True # or False

    VIDEO_SOURCE = 7  # Webcam or "path/to/your/video.mp4"
    # VIDEO_SOURCE = "your_test_video.mp4"

    import os
    try:
        source_int = int(VIDEO_SOURCE)
        cap = cv2.VideoCapture(source_int)
    except ValueError:
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Test Error: Could not open video source '{VIDEO_SOURCE}'.")
        exit()
    print(f"Test: Opened video source: {VIDEO_SOURCE}")

    cv2.namedWindow('Stop Utils Test Output', cv2.WINDOW_NORMAL)
    frame_count = 0
    total_proc_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if not isinstance(VIDEO_SOURCE, int): # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
                if not ret: break
            else: break # Webcam error
        if frame is None or frame.size == 0: continue

        loop_start = time.time()
        
        # --- Call the analysis function ---
        # It returns: (bool_detected, frame_output_from_analysis)
        # frame_output_from_analysis is either original or drawn upon based on SHOW_INTERNAL_VISUALS
        stop_detected, displayed_frame = analysis(frame)

        proc_time = time.time() - loop_start
        total_proc_time += proc_time
        frame_count += 1
        fps = 1.0 / proc_time if proc_time > 0 else 0

        if stop_detected:
            print(f"Frame {frame_count}: Stop sign DETECTED! (FPS: {fps:.1f})")
        
        # The displayed_frame already has drawings if SHOW_INTERNAL_VISUALS=True and sign found
        cv2.putText(displayed_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Add FPS to final displayed frame
        cv2.imshow('Stop Utils Test Output', displayed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_count > 0:
        avg_fps = frame_count / total_proc_time if total_proc_time > 0 else 0
        print(f"\nTest Summary: Processed {frame_count} frames. Avg FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Test: Exiting stop_utils test.")

