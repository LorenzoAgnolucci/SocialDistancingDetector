from camera_calibration import get_calibrated_image, get_bird_view
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 1 meter away
    global mouseX, mouseY, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 5, (0, 255, 255), 5)
        if "mouse_pts" not in globals():
            mouse_points = []
        mouse_points.append((x, y))
        print("Point detected")
        print(mouse_points)


def check_in_ROI(point, ROI):
    return (ROI[0][0] <= point[0] <= ROI[1][0]) and (ROI[0][1] <= point[1] <= ROI[2][1])


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
    return results_labels, violating_pairs


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

video_stream = cv2.VideoCapture(0)
height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video_stream.get(cv2.CAP_PROP_FPS))

frame_num = 0
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
mouse_points = []
homography_matrix = []
warped_ROI_points = []
distance_threshold = 0

while video_stream.isOpened():
    frame_num += 1
    ret, frame = video_stream.read()
    frame = get_calibrated_image(frame, CALIBRATION_MATRIX_PATH)

    if not ret:
        print("End of the video file")
        break

    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    if frame_num == 1:
        while len(mouse_points) <= 5:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
        cv2.destroyWindow("image")
        # mouse_points[:4] -> ROI (from top left clockwise)  mouse_points[4:6] -> distance points (1 meter)
        homography_matrix = get_bird_view(mouse_points[:4], frame)

        mouse_points = np.array(mouse_points).reshape(1, -1, 2).astype(np.float32)
        warped_mouse_points = cv2.perspectiveTransform(mouse_points, homography_matrix)[0]

        warped_ROI_points = warped_mouse_points[:4]
        warped_distance_points = warped_mouse_points[4:6]

        distance_threshold = np.sqrt((warped_distance_points[0][0] - warped_distance_points[1][0]) ** 2
                                     + (warped_distance_points[0][1] - warped_distance_points[1][1]) ** 2)

    # frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, 0)
    violating_pairs = []
    if results:
        results_labels, violating_pairs = compute_distances(results, warped_ROI_points, homography_matrix, distance_threshold)
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid

            color = (0, 255, 0)
            if results_labels[i] == -1:
                color = (0, 255, 255)
            elif results_labels[i] == 1:
                color = (0, 0, 255)
                for j in [el[1] for el in violating_pairs if el[0] == i]:
                    (jcX, jcY) = results[j][2]
                    cv2.line(frame, (cX, cY), (jcX, jcY), color, 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

    text = "Social Distancing Violations: {}".format(len(violating_pairs))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    cv2.imshow("Video stream", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    print("Processing frame: ", frame_num)
