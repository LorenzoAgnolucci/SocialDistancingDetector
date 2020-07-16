from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
from camera_calibration import get_calibrated_image
import numpy as np
import cv2
import os


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 1 meter away
    global mouseX, mouseY, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 5, (0, 255, 255), 5)
        if "mouse_points" not in globals():
            mouse_points = []
        mouse_points.append((x, y))
        print("Point detected")
        print(mouse_points)


def check_in_ROI(point, ROI):
    return (ROI[0][0] <= point[0] <= ROI[1][0]) and (ROI[0][1] <= point[1] <= ROI[2][1])


def get_bird_view(original_ROI_points, img):
    AREA_THRESHOLD = 100000
    AREA_SCALE_FACTOR = 1.15

    original_ROI_points = np.array(original_ROI_points)

    bbox_width = int(np.sqrt(((original_ROI_points[0][0]-original_ROI_points[1][0])**2) + ((original_ROI_points[0][1]-original_ROI_points[1][1])**2)))
    bbox_height = int(np.sqrt(((original_ROI_points[2][0]-original_ROI_points[1][0])**2) + ((original_ROI_points[2][1]-original_ROI_points[1][1])**2)))

    while bbox_height*bbox_width <= AREA_THRESHOLD:
        bbox_height = int(bbox_height*AREA_SCALE_FACTOR)
        bbox_width = int(bbox_width*AREA_SCALE_FACTOR)
    aspect_ratio = bbox_width / bbox_height
    img_h, img_w = img.shape[:2]
    if aspect_ratio <= 1:
        bbox_width = int(img_h * aspect_ratio)
        bbox_height = img_h
    else:
        bbox_width = img_w
        bbox_height = int(img_w / aspect_ratio)
    bbox = [[0, 0], [bbox_width, 0], [bbox_width, bbox_height], [0, bbox_height]]
    bird_view_ROI_points = np.array(bbox)

    homography_matrix, status = cv2.findHomography(original_ROI_points, bird_view_ROI_points)

    # Warp source image to destination based on homography
    # img_out = cv2.warpPerspective(img, homography_matrix, (bbox_width, bbox_height))

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

    CALIBRATION_MATRIX_PATH = 'calibration_matrix.yml'

    # video_stream = cv2.VideoCapture(0)
    video_stream = cv2.VideoCapture('video/pedestrians.mp4')

    # height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fps = int(video_stream.get(cv2.CAP_PROP_FPS))

    frame_num = 0
    cv2.namedWindow("First Frame")
    cv2.setMouseCallback("First Frame", get_mouse_points)
    mouse_points = []
    homography_matrix = []
    warped_ROI_points = []
    distance_threshold = 0

    while video_stream.isOpened():

        frame_num += 1
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (720, 480))
        frame = get_calibrated_image(frame, CALIBRATION_MATRIX_PATH)

        if not ret:
            print("End of the video file")
            break

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        if frame_num == 1:
            while len(mouse_points) <= 5:
                image = frame
                cv2.imshow("First frame", image)
                cv2.waitKey(1)
            cv2.destroyWindow("First frame")
            # mouse_points[:4] -> ROI (from top left clockwise)  mouse_points[4:6] -> distance points (1 meter)
            homography_matrix, aspect_ratio = get_bird_view(mouse_points[:4], frame)

            warped_mouse_points = cv2.perspectiveTransform(np.array(mouse_points).reshape(1, -1, 2).astype(np.float32), homography_matrix)[0]
            warped_ROI_points = warped_mouse_points[:4]
            warped_distance_points = warped_mouse_points[4:6]

            distance_threshold = np.sqrt((warped_distance_points[0][0] - warped_distance_points[1][0]) ** 2
                                         + (warped_distance_points[0][1] - warped_distance_points[1][1]) ** 2)

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
                        cv2.line(bird_view, (warpedX, warpedY), (jwarpedX, jwarpedY), color, 1)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                if results_labels[i] != -1:
                    cv2.circle(bird_view, (warpedX, warpedY), int(distance_threshold / 2), color, 1)
                    cv2.circle(bird_view, (warpedX, warpedY), 1, color, 1)

        cv2.polylines(frame, np.int32([mouse_points[:4]]), True, (168, 50, 124), 2)

        border_text = "Social Distancing Violations: {}".format(len(violating_pairs))
        cv2.putText(frame, border_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        text = "Social Distancing Violations: {}".format(len(violating_pairs))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        text_bv = "Bird View"
        cv2.putText(bird_view, text_bv, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if aspect_ratio <= 1:
            split_image = np.hstack((frame, bird_view))
        else:
            split_image = np.vstack((frame, bird_view))

        cv2.imshow("Social Distancing Detector", split_image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # print("Processing frame: ", frame_num)
