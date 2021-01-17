import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image

w = 7
h = 5
class StereoCalibration(object):
    def __init__(self, filepath):
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.objp = np.zeros((w * h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)
        print(self.cal_path)


    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + 'Right/*.jpg')
        images_left = glob.glob(cal_path + 'Left/*.jpg')
        images_left.sort()
        images_right.sort()

        print("9")
        for i, fname in enumerate(images_right):
            print(i)
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])


            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (w, h), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (w, h), None)

            self.objpoints.append(self.objp)
            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
            self.imgpoints_l.append(corners_l)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
            self.imgpoints_r.append(corners_r)

            img_shape = gray_l.shape[::-1]
            #print("1")
            rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_l, img_shape, None, None)
            rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_r, img_shape, None, None)


           # print("0")
        self.camera_model = self.stereo_calibrate(img_shape)
            #print(self.camera_model)




    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        rectify_scale = 1  # 设置为0的话，对图片进行剪裁，设置为1则保留所有原图像像素
        ChessImaR = cv2.imread('E:/Image/Right/R_2.jpg', 0)
        ChessImaL = cv2.imread('E:/Image/Right/R_2.jpg', 0)
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(M1, d1, M2, d2,
                                                          ChessImaR.shape[::-1], R, T,
                                                          rectify_scale, (0, 0))
        Left_Stereo_Map = cv2.initUndistortRectifyMap(M1, d1, RL, PL,
                                                      ChessImaR.shape[::-1], cv2.CV_16SC2)

        Right_Stereo_Map = cv2.initUndistortRectifyMap(M2, d2, RR, PR,
                                                       ChessImaR.shape[::-1], cv2.CV_16SC2)

        # 立体校正效果显示
        for i in range(0, 1):  # 以第一对图片为例
            t = str(i)
            frameR = cv2.imread('E:/Image/Right/R_2.jpg', 0)
            frameL = cv2.imread('E:/Image/Right/R_2.jpg', 0)

            Left_rectified = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                       cv2.BORDER_CONSTANT, 0)  # 使用remap函数完成映射
            im_L = Image.fromarray(Left_rectified)  # numpy 转 image类

            Right_rectified = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                        cv2.BORDER_CONSTANT, 0)
            im_R = Image.fromarray(Right_rectified)  # numpy 转 image 类

            # 创建一个能同时并排放下两张图片的区域，后把两张图片依次粘贴进去
            width = im_L.size[0] * 2
            height = im_L.size[1]

            img_compare = Image.new('RGBA', (width, height))
            img_compare.paste(im_L, box=(0, 0))
            img_compare.paste(im_R, box=(1441, 0))

            # 在已经极线对齐的图片上均匀画线
            for i in range(1, 20):
                len = 480 / 20
                plt.axhline(y=i * len, color='r', linestyle='-')
            plt.imshow(img_compare)
            plt.show()

        return camera_model

cal = StereoCalibration('E:/Image/')
cal.camera_model
