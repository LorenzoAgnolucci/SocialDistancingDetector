import numpy as np
import cv2
import glob


def calibrate_camera():
    """Calibrate the camera with the given images and return the matrices"""
    CHESSBOARD_WIDTH = 9
    CHESSBOARD_HEIGHT = 6
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_HEIGHT * CHESSBOARD_WIDTH, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(IMAGES_PATH)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, - 1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners2, ret)
            cv2.imshow(fname, img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    return mtx, dist


def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)

    cv_file.release()


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def get_calibrated_image(image, matrix_path):
    if matrix_path == "":
        return image

    mtx, dist = load_coefficients(matrix_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    calibrated_image = cv2.undistort(image, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    calibrated_image = calibrated_image[y:y + h + 1, x:x + w + 1]
    return calibrated_image


if __name__ == '__main__':
    IMAGES_PATH = f"calibration_images/webcam_images/*.jpg"
    MATRIX_PATH = 'webcam_calibration_matrix.yml'
    mtx, dist = calibrate_camera()
    save_coefficients(mtx, dist, MATRIX_PATH)
    img = cv2.imread("calibration_images/webcam_images/image11.jpg")
    cal = get_calibrated_image(img, MATRIX_PATH)
    cv2.imshow("img", img)
    cv2.imshow("cal", cal)
    cv2.waitKey(0)
