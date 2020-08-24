from object_detection import neural_network_config as config
from object_detection.object_detection import detect_people
from scipy.spatial import distance as dist
import math
from camera_calibration import get_calibrated_image
import numpy as np
import cv2
import os
from VideoGet import VideoGet
from imutils.video import FPS
import streamlink


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the first frame of the video that will be warped
    # Used to mark 2 points on the first frame of the video that are 1 meter away
    global mouseX, mouseY, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(frame, (x, y), 5, (0, 255, 255), 5)
        if "mouse_points" not in globals():
            mouse_points = []
        mouse_points.append((x, y))
        print("Point detected")
        print(mouse_points)


def check_in_ROI(point, ROI):
    return (ROI[0][0] <= point[0] <= ROI[1][0]) and (ROI[0][1] <= point[1] <= ROI[2][1])


def get_bird_view(original_ROI_points, img):
    img_height, img_width = img.shape[:2]

    # image center
    u0 = img_width / 2
    v0 = img_height / 2

    # detected corners on the original image
    p = [original_ROI_points[0], original_ROI_points[1], original_ROI_points[3], original_ROI_points[2]]

    # widths and heights of the projected image
    w1 = dist.euclidean(p[0], p[1])
    w2 = dist.euclidean(p[2], p[3])

    h1 = dist.euclidean(p[0], p[2])
    h2 = dist.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal distance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
                n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # compute the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    aspect_ratio = W / H

    if aspect_ratio <= 1:
        bbox_width = int(img_height * aspect_ratio)
        bbox_height = img_height
    else:
        bbox_width = img_width
        bbox_height = int(img_width / aspect_ratio)
    bbox = [[0, 0], [bbox_width, 0], [bbox_width, bbox_height], [0, bbox_height]]

    original_ROI_points = np.array(original_ROI_points).astype('float32')
    bird_view_ROI_points = np.float32(bbox)

    # project the image with the new w/h
    homography_matrix, status = cv2.findHomography(original_ROI_points, bird_view_ROI_points)

    img_out = cv2.warpPerspective(img, homography_matrix, (bbox_width, bbox_height))
    cv2.imshow("Warped", img_out)
    cv2.waitKey(0)

    return homography_matrix, aspect_ratio


def compute_distances(results, warped_ROI_points, homography_matrix, distance_threshold):
    results_feet_points = [(r[2][0], r[1][3]) for r in results]     # x of centroid, y of bottom right of the bbox
    results_feet_points = np.array(results_feet_points)
    results_feet_points = results_feet_points.reshape(1, -1, 2).astype(np.float32)
    warped_results_feet_points = cv2.perspectiveTransform(results_feet_points, homography_matrix)[0]

    results_labels = [0] * len(results)  # -1 yellow (out_of_ROI), 0 green (no_violation), 1 red (violation)
    distances = dist.cdist(warped_results_feet_points, warped_results_feet_points, metric='euclidean')
    violating_pairs = []
    for i in range(0, distances.shape[0]):
        if not check_in_ROI(warped_results_feet_points[i], warped_ROI_points):
            results_labels[i] = -1
            continue
        for j in range(i + 1, distances.shape[1]):
            if not check_in_ROI(warped_results_feet_points[j], warped_ROI_points):
                continue
            if distances[i][j] <= distance_threshold:
                violating_pairs.append((i, j))
                results_labels[i] = results_labels[j] = 1
    return results_labels, violating_pairs, warped_results_feet_points


if __name__ == '__main__':
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    WEBCAM_CALIBRATION_MATRIX_PATH = 'calibration_matrix.yml'
    PHONE_CALIBRATION_MATRIX_PATH = 'phone_calibration_matrix.yml'
    DISTANCE_THRESHOLD_METERS = 2

    # stream_source = 0     # Computer webcam
    # stream_source = 'https://192.168.1.253:8080/video'     # Phone camera stream
    isPhone = False
    # isPhone = True
    # video_getter = VideoGet(stream_source, True).start()

    # stream_source = streamlink.streams('https://www.youtube.com/watch?v=srlpC5tmhYs')['best'].url     # Remote youtube live stream
    stream_source = 'video/pedestrians.mp4'       # Local video
    video_getter = VideoGet(stream_source, False).start()

    fps = FPS().start()

    frame_num = 0
    cv2.namedWindow("First frame")
    cv2.setMouseCallback("First frame", get_mouse_points)
    mouse_points = []
    homography_matrix = []
    warped_ROI_points = []
    distance_threshold = 0
    cumulative_total_people = 0
    cumulative_violating_people = 0

    while video_getter.more():

        frame_num += 1
        frame = video_getter.read()

        if stream_source == 0:
            frame = get_calibrated_image(frame, WEBCAM_CALIBRATION_MATRIX_PATH)
        if isPhone:
            frame = get_calibrated_image(frame, PHONE_CALIBRATION_MATRIX_PATH)

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        video_stream_aspect_ratio = frame_w / frame_h

        frame = cv2.resize(frame, (int(video_stream_aspect_ratio*600), 600))

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        if frame_num == 1:
            while len(mouse_points) <= 5:
                text = "1) Insert 4 (rectangular in real world) ROI points from top-left in clockwise order"
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                text = "2) Insert 2 points on the ROI plane corresponding to 1 meter in real world"
                cv2.putText(frame, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                text = "1) Insert 4 (rectangular in real world) ROI points from top-left in clockwise order"
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                text = "2) Insert 2 points on the ROI plane corresponding to 1 meter in real world"
                cv2.putText(frame, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                cv2.imshow("First frame", frame)
                cv2.waitKey(1)
            cv2.destroyWindow("First frame")
            # mouse_points[:4] -> ROI (from top left clockwise)  mouse_points[4:6] -> distance points (1 meter)
            homography_matrix, aspect_ratio = get_bird_view(mouse_points[:4], frame)

            warped_mouse_points = cv2.perspectiveTransform(np.array(mouse_points).reshape(1, -1, 2).astype(np.float32), homography_matrix)[0]
            warped_ROI_points = warped_mouse_points[:4]
            warped_distance_points = warped_mouse_points[4:6]

            distance_threshold = np.sqrt((warped_distance_points[0][0] - warped_distance_points[1][0]) ** 2
                                         + (warped_distance_points[0][1] - warped_distance_points[1][1]) ** 2)
            distance_threshold = distance_threshold * DISTANCE_THRESHOLD_METERS

        bird_view = []
        if aspect_ratio <= 1:
            bird_view = np.zeros((frame_h, int(frame_h * aspect_ratio), 3), np.uint8)
            bird_view[:] = (41, 41, 41)
        else:
            bird_view = np.zeros((int(frame_w / aspect_ratio), frame_w, 3), np.uint8)
            bird_view[:] = (41, 41, 41)

        results = detect_people(frame, net, ln, 0)
        violating_pairs = []
        if results:
            results_labels, violating_pairs, warped_results_feet_points = compute_distances(results, warped_ROI_points, homography_matrix, distance_threshold)
            for (i, (prob, bbox, centroid)) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid

                warpedX, warpedY = warped_results_feet_points[i]

                color = (0, 255, 0)
                if results_labels[i] == -1:
                    color = (0, 255, 255)
                elif results_labels[i] == 1:
                    color = (0, 0, 255)

                    for j in [el[1] for el in violating_pairs if el[0] == i]:
                        (jcX, jcY) = results[j][2]
                        (jwarpedX, jwarpedY) = warped_results_feet_points[j]
                        cv2.line(frame, (cX, cY), (jcX, jcY), color, 2)
                        cv2.line(bird_view, (int(warpedX), int(warpedY)), (int(jwarpedX), int(jwarpedY)), color, 1)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                if results_labels[i] != -1:
                    cv2.circle(bird_view, (int(warpedX), int(warpedY)), int(distance_threshold / 2), color, 1)
                    cv2.circle(bird_view, (int(warpedX), int(warpedY)), 1, color, 1)

        cv2.polylines(frame, np.int32([mouse_points[:4]]), True, (168, 50, 124), 2)

        current_total_people = len([el for el in results_labels if el == 1 or el == 0])
        current_violating_people = len([el for el in results_labels if el == 1])
        if current_total_people:
            current_violating_percentage = format(current_violating_people / current_total_people * 100, ".1f")
        else:
            current_violating_percentage = 0.0

        cumulative_total_people += current_total_people
        cumulative_violating_people += current_violating_people
        if cumulative_total_people:
            cumulative_violating_percentage = format(cumulative_violating_people / cumulative_total_people * 100, ".1f")
        else:
            cumulative_violating_percentage = 0.0

        current_border_text = f"Current Social Distancing Violations: {current_violating_percentage}%"
        cv2.putText(frame, current_border_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        current_text = f"Current Social Distancing Violations: {current_violating_percentage}%"
        cv2.putText(frame, current_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cumulative_border_text = f"Cumulative Social Distancing Violations: {cumulative_violating_percentage}%"
        cv2.putText(frame, cumulative_border_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cumulative_text = f"Cumulative Social Distancing Violations: {cumulative_violating_percentage}%"
        cv2.putText(frame, cumulative_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        text_bv = "Bird View"
        cv2.putText(bird_view, text_bv, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if aspect_ratio <= 1:
            split_image = np.hstack((frame, bird_view))
        else:
            split_image = np.vstack((frame, bird_view))

        cv2.imshow("Social Distancing Detector", split_image)
        cv2.moveWindow("Social Distancing Detector", 65, 20)
        key = cv2.waitKey(1) & 0xFF
        fps.update()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    fps.stop()
    print(f"FPS: {fps.fps()}")
    print(f"Elapsed: {fps.elapsed()}")
    cv2.destroyAllWindows()
    video_getter.stop()
