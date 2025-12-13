import cv2
import numpy as np
from ultralytics import YOLO
from shared_objects.ROS_utils import Topics, SHOW
import time, torch

device = torch.device('cuda')
topics=Topics()
topic_names=topics.topic_names
row, col, overlap = 5, 5, 0.3
img_size = (640, 360)
confidence_model = 0.2
confidence_verifier = 0.5
threshold = None


# creating yolo model
model = YOLO("/home/bylogix/Downloads/stop_weights.pt").to(device)
verifier_model= YOLO("/home/bylogix/Downloads/yolo12x.pt").to(device)

def slice_frame(frame, rows, cols, overlap):
    """ takes the data for the slicing from the global variables 
    and cuts in that number of images the arleady cut(ted?) image"""
    height, width, _ = frame.shape
    step_height = height // rows
    step_width = width // cols

    # Calculate the overlap in pixels
    overlap_height = int(step_height * overlap)
    overlap_width = int(step_width * overlap)
    
    slices = []
    offsetsr = []
    for i in range(rows):
        for j in range(cols):
            y1 = max(0, step_height * i - overlap_height)
            y2 = min(height, step_height * (i + 1) + overlap_height)
            x1 = max(0, step_width * j - overlap_width)
            x2 = min(width, step_width * (j + 1) + overlap_width)

            slice_img = frame[y1:y2, x1:x2]
            slices.append(slice_img) 
            offsetsr.append((x1,y1)) # track offset of slice

    return slices, offsetsr
"""
    return (frame[max(0, step_height*i - overlap_height):min(height, step_height*(i+1) + overlap_height),
                  max(0, step_width*j - overlap_width):min(width, step_width*(j+1) + overlap_width)]
            for i in range(rows) for j in range(cols))
"""
def analysis(frame):
    # Slice the frame into 12 pieces (3 rows x 4 columns)
    sliced_frames, offsets = list(slice_frame(frame, row, col, overlap))
    start=time.time()
    # we give our results to yolo
    results = model.predict(sliced_frames, save=True, imgsz=640,conf=confidence_model)#, show_labels=True, show_conf=True, show_boxes=True)
    print(f"result took {time.time()-start}")
    # Calculate the width and height of each slice
    slice_height, slice_width = frame.shape[0] // row, frame.shape[1] // col

    if results is not None:
        for num, result in enumerate(results):
            colors = np.random.randint(0, 255, size=(len(result.boxes.conf), 3), dtype=np.uint8)
            for i in range(len(result.boxes.conf)):
                xy = result.boxes.xyxy[i]
                xy_np = xy.cpu().numpy()
                x1, y1, x2, y2 = map(int, xy_np)
                confidence = result.boxes.conf[i]
                label = result.names[int(result.boxes.cls[i])]
                if label == "stop sign":
                    offset_x, offset_y = offsets[num][0], offsets[num][1]
                    x1 += offset_x
                    y1 += offset_y
                    x2 += offset_x
                    y2 += offset_y
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    verify_results = verifier_model.predict(cropped, imgsz=320, conf=confidence_verifier)
                    stop_sign_verified = False
                    for verify_result in verify_results:
                        for j in range(len(verify_result.boxes)):
                            cls_id = int(verify_result.boxes.cls[j])
                            #name = verify_result.names[cls_id]
                            if cls_id == 11:
                                stop_sign_verified = True
                                break
                        if stop_sign_verified:
                            break

                    if not stop_sign_verified:
                        continue
                    if SHOW:
                            # Calculate the offset based on the slice's position, warning at col or row
                        offset_x, offset_y = (num % col) * slice_width, (num // col) * slice_height

                            # Adjust the bounding box coordinates
                        x1 += offset_x
                        y1 += offset_y
                        x2 += offset_x
                        y2 += offset_y

                            # Set the position for the label text
                        label_position = (x1, y1 - 10) # a little bit above

                            # Set the font, font scale, and thickness
                        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 3

                            # Use a different color for each box
                        color = tuple(map(int, colors[i]))

                            # Draw the bounding box
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        label_text = f"{label}: {confidence:.2f}"
                        print(f"Labels: {label_position}\nBoxes conf: {results.boxes.conf}")
                            # Put the label text on the image
                        frame = cv2.putText(frame, label_text, label_position, font, font_scale, color, thickness)
                            # Display the image with bounding boxes and labels
                        cv2.imshow('All for one, one for all', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            break
                    return [True, frame]
    return [False, frame]
