import os
import cv2 as cv
import numpy as np


clicked_points = []
original_image = None
display_image = None


calculated_M = None
waiting_for_save = False


def get_vanishing_point(p1, p2, p3, p4):
    l1 = np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
    l2 = np.cross([p3[0], p3[1], 1], [p4[0], p4[1], 1])
    v = np.cross(l1, l2)

    if v[2] == 0:
        return None
    return (int(v[0] / v[2]), int(v[1] / v[2]))


def get_x_from_y(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2
    if y1 == y2:
        return x1
    return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))


def apply_birdseye_view(vp, left_line, right_line, img):
    h, w = img.shape[:2]

    top_y = vp[1] + 50
    bottom_y = h

    left_top_x = get_x_from_y(left_line[0], left_line[1], top_y)
    left_bottom_x = get_x_from_y(left_line[0], left_line[1], bottom_y)

    right_top_x = get_x_from_y(right_line[0], right_line[1], top_y)
    right_bottom_x = get_x_from_y(right_line[0], right_line[1], bottom_y)

    if left_bottom_x > right_bottom_x:
        left_top_x, right_top_x = right_top_x, left_top_x
        left_bottom_x, right_bottom_x = right_bottom_x, left_bottom_x

    print("\n--- Calculated Transformation Points ---")
    print(
        f"Left Top: ({left_top_x}, {top_y}), Right Top: ({right_top_x}, {top_y})")
    print(
        f"Left Bottom: ({left_bottom_x}, {bottom_y}), Right Bottom: ({right_bottom_x}, {bottom_y})")

    src = np.float32([
        [left_top_x, top_y],
        [right_top_x, top_y],
        [right_bottom_x, bottom_y],
        [left_bottom_x, bottom_y]
    ])

    dst_margin = 300
    dst = np.float32([
        [dst_margin, 0],
        [w - dst_margin, 0],
        [w - dst_margin, h],
        [dst_margin, h]
    ])

    print(f"src matrix = \n{src}")
    print(f"dst matrix = \n{dst}")

    M = cv.getPerspectiveTransform(src, dst)
    bev_image = cv.warpPerspective(img, M, (w, h))

    pts = src.reshape((-1, 1, 2)).astype(np.int32)
    cv.polylines(display_image, [pts], isClosed=True,
                 color=(0, 255, 0), thickness=2)

    return bev_image, M


def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_image, original_image
    global calculated_M, waiting_for_save

    # Don't accept new clicks if we are already waiting for save input
    if waiting_for_save:
        return

    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            cv.circle(display_image, (x, y), 5, (0, 255, 0), -1)

            if len(clicked_points) == 2:
                cv.line(display_image,
                        clicked_points[0], clicked_points[1], (255, 0, 0), 2)

            elif len(clicked_points) == 4:
                cv.line(display_image,
                        clicked_points[2], clicked_points[3], (255, 0, 0), 2)

                vp = get_vanishing_point(
                    clicked_points[0], clicked_points[1],
                    clicked_points[2], clicked_points[3]
                )

                if vp:
                    print(f"\nVanishing Point found at: {vp}")
                    cv.circle(display_image, vp, 8, (0, 0, 255), -1)

                    left_line = (clicked_points[0], clicked_points[1])
                    right_line = (clicked_points[2], clicked_points[3])
                    bev_result, M = apply_birdseye_view(
                        vp, left_line, right_line, original_image)

                    cv.imshow("Original Setup", display_image)
                    cv.namedWindow("Bird's Eye View", cv.WINDOW_NORMAL)
                    cv.resizeWindow("Bird's Eye View", 800, 600)
                    cv.imshow("Bird's Eye View", bev_result)

                    # Instead of blocking here, we just set the flags!
                    calculated_M = M
                    waiting_for_save = True
                    print(
                        "\n>>> PRESS 'y' TO SAVE CONFIGURATION, OR 'n' TO EXIT INSIDE OF THE PROGRAM <<<")

                else:
                    print("Lines are perfectly parallel; no vanishing point.")
                    clicked_points.clear()  # reset to try again

            cv.imshow("Original Setup", display_image)


image_path = os.path.expanduser(
    "/home/ubuntu/Workspace/ros-bridge/src/carla_ros_bridge/calibration_setup/frame_0.png")
original_image = cv.imread(image_path)

if original_image is None:
    raise ValueError(f"cant find the image {image_path}")

display_image = original_image.copy()

window_name = "Original Setup"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 800, 600)
cv.setMouseCallback(window_name, mouse_callback)

print("Instructions:")
print("1. Click two points along the LEFT lane line.")
print("2. Click two points along the RIGHT lane line.")
print("Image will be warped after the 4th point")
cv.imshow(window_name, display_image)

while True:
    key = cv.waitKey(30) & 0xFF

    if waiting_for_save:
        if key == ord('y'):
            np.save("/home/ubuntu/Workspace/ros-bridge/src/carla_ros_bridge/calibration_setup/bev_matrix.npy", calculated_M)
            print("Saved to bev_matrix.npy. Exiting.")
            break
        elif key == ord('n') or key == 27:  # 27 is ESC
            print("Configuration not saved. Exiting.")
            break
    else:
        # Allow exiting before clicking 4 times
        if key == 27:
            print("Exiting.")
            break

cv.destroyAllWindows()
