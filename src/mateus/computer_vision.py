import rospy
import numpy as np
import cv2 as cv
import scipy as sci
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as rot
from cv_bridge import CvBridge, CvBridgeError


class ComputerVision():

    def __init__(self):

        self.rate = rospy.Rate(100)

        self.camera_subscriber = rospy.Subscriber('/camera/camera1/image_raw', Image, self.callback_image, queue_size=1)

        self.camera_info_sub = rospy.Subscriber('/camera/camera1/camera_info', CameraInfo, self.callback_camerainfo, queue_size=1)

        self.bridge = CvBridge()

        self.image = None
    
    def callback_image(self, data):
        """
        Recovers front image from the camera mounted on the robot
        """

        # print(data)

        self.data = data.height

        self.image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    
    def callback_camerainfo(self, data):

        K = data.K
        D = data.D

        self.mtx = np.array([[K[0], K[1], K[2]], [K[3], K[4], K[5]], [K[6], K[7], K[8]]])
        self.dist = np.array([[D[0], D[1], D[2], D[3], D[4]]])


    def aruco_detection(self, image, dictionary, rvecs, tvecs):
        
        position = None
        q_obj_b = None
        cam_vec = None
        camera_meas_flag = None

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        #Initialize the detector parameters using defaults values
        parameters = cv.aruco.DetectorParameters_create()
        parameters.adaptiveThreshWinSizeMin = 10
        parameters.adaptiveThreshWinSizeMax = 20
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.adaptiveThreshConstant = 10

        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
        parameters.cornerRefinementMaxIterations = 5
        # parameters.cornerRefinementMinAccuracy = 0.01


        #Detect the markers in the image
        markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        

        #If there is a marker compute the pose
        if markerIDs is not None and len(markerIDs)==2:

            #Descending order Marker ID's array
            if len(markerIDs)==2:
                order = np.argsort(-markerIDs.reshape(1,2))
                markerIDs = markerIDs[order].reshape(2,1)
                markerCorners = np.asarray(markerCorners, dtype='float32')[order]

            for i in range(0, len(markerIDs)):
                
                #Check if it's reference marker
                if markerIDs[i]==10:
                    
                    #Compute Pose Estimation of the Reference Marker
                    rvec_ref, tvec_ref, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.2, self.mtx, self.dist, rvecs, tvecs)
                    rvec_ref = np.reshape(rvec_ref, (3,1))
                    tvec_ref = np.reshape(tvec_ref, (3,1))
                    #Use Rodrigues formula to transform rotation vector into matrix
                    #Pose marker w.r.t camera reference frame
                    R_rc, _ = cv.Rodrigues(rvec_ref)
                    #Homogeneous Transformation Fixed Frame to Camera Frame
                    last_col = np.array([[0, 0, 0, 1]])
                    T_rc = np.concatenate((R_rc, tvec_ref), axis=1)
                    T_rc = np.concatenate((T_rc, last_col), axis=0)
                    #Homegeneous Transformation Camera Frame to Fixed Frame
                    T_cr = np.linalg.inv(T_rc)

                    #Get Reference's Marker attitude w.r.t camera
                    r_ref = sci.spatial.transform.Rotation.from_matrix(T_cr[0:3, 0:3])
                    q_ref = r_ref.as_quat()
                    euler_ref = r_ref.as_euler('XYZ')

                    #Draw axis in marker
                    cv.aruco.drawAxis(image, self.mtx, self.dist, rvec_ref, tvec_ref, 0.2)
                    cv.aruco.drawDetectedMarkers(image, markerCorners[0])

                #Check if there is moving marker/object marker
                if markerIDs[i]==4:
                

                    #Get Pose Estimation of the Moving Marker
                    rvec_obj, tvec_obj, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.12, self.mtx, self.dist, rvecs, tvecs)
                    rvec_obj = np.reshape(rvec_obj, (3,1))
                    tvec_obj = np.reshape(tvec_obj, (3,1))

                    #Use Rodrigues formula to transform rotation vector into matrix
                    R_mc, _ = cv.Rodrigues(rvec_obj)

                    #Homogeneous Transformation Object Frame to Camera Frame
                    last_col = np.array([[0, 0, 0, 1]])
                    T_mc = np.concatenate((R_mc, tvec_obj), axis=1)
                    T_mc = np.concatenate((T_mc, last_col), axis=0)

                    r_mov = sci.spatial.transform.Rotation.from_matrix(T_mc[0:3, 0:3])
                    q_mov = r_mov.as_quat()
                    euler_mov = r_mov.as_euler('XYZ')
                    

                    if T_cr is not None:
                        #Homogeneous Transformation Object Frame to Fixed Frame
                        T_mr = T_cr@T_mc
                    else:
                        T_mr = np.eye(4)@T_mc

                    T_rm = T_mr.T

                    #Getting quaternions from rotation matrix
                    r_obj = sci.spatial.transform.Rotation.from_matrix(T_mr[0:3, 0:3])
                    q_obj = r_obj.as_quat()
                    euler_obj = r_obj.as_euler('XYZ')
                    
                    #Getting quaternion from Fixed Frame to Body Frame
                    r_obj_b = sci.spatial.transform.Rotation.from_matrix(T_rm[0:3, 0:3])
                    q_obj_b = r_obj_b.as_quat()


                    #Marker's Position
                    zf_obj = float(T_mr[2,3])
                    xf_obj = float(T_mr[0,3])
                    yf_obj = float(T_mr[1,3])

                    position = np.array([[xf_obj, yf_obj, zf_obj]]).T

                    #Measurement Direction Vector
                    
            
                    cx = float(q_obj_b[3])**2 + float(q_obj_b[0])**2 - float(q_obj_b[1])**2 - float(q_obj_b[2])**2
                    cy = 2*(float(q_obj_b[0])*float(q_obj_b[1]) + float(q_obj_b[3])*float(q_obj_b[2]))
                    cz = 2*(float(q_obj_b[0])*float(q_obj_b[2]) - float(q_obj_b[3])*float(q_obj_b[1]))


                    cam_vec = np.array([[cx, cy, cz]]).T
                    cam_vec *= 1/(np.linalg.norm(cam_vec))
                                     

                    #Draw ArUco contourn and Axis
                    cv.aruco.drawAxis(image, self.mtx, self.dist, rvec_obj, tvec_obj, 0.12)
                    cv.aruco.drawDetectedMarkers(image, markerCorners[0])

        return position, q_obj_b, cam_vec, image
    


    def img_show(self):

        rvecs = None
        tvecs = None

        rvecs2 = None
        tvecs2 = None

        x_Fold = 0
        y_Fold = 0
        z_Fold = 0
        
        global T_cr

        # T_cr = np.eye(4)

        #Font setup
        font = cv.FONT_HERSHEY_PLAIN
        
        #Load the predefinied dictionary
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

        
        
        
        ####################################### ARUCO MARKER POSE ESTIMATION #########################################################################


        self.pos_cam, self.quat, self.cam_vec, self.image = self.aruco_detection(self.image, dictionary, rvecs, tvecs)
                

        # self.position2, self.quat2, self.cam_vec2, image2 = self.aruco_detection(image2, dictionary, rvecs2, tvecs2)



        # if self.cam_vec is not None and self.cam_vec_mean is not None:
    

        #     error_cam_vec = abs(self.cam_vec - self.cam_vec_mean)

        #     if (error_cam_vec > 0.05).any():

        #         self.cam_vec = None
        #         self.cam_vec_off = True
        #     else:
        #         self.cam_vec_off = False

        # if self.cam_vec2 is not None and self.cam_vec2_mean is not None:
    

        #     error_cam_vec2 = abs(self.cam_vec2 - self.cam_vec2_mean)

        #     if (error_cam_vec2 > 0.05).any():

        #         self.cam_vec2 = None
        #         self.cam_vec2_off = True
        #     else:
        #         self.cam_vec2_off = False



        # self.cam_vec, self.cam_vec2 = None, None

        # self.position = None
        # self.position2 = None

        # if self.position is not None:
        #     #Corrected Position Camera 1

        #     self.position[0] = self.position[0]*0.968 + 0.0845
        #     self.position[1] = self.position[1]*0.895 + 0.632
        #     self.position[2] = self.position[2]*0.83 - 1.71

        #     # Z correction
        #     self.position[0] -= -self.position[2]*0.0299 + 0.0559
        #     self.position[1] -= -self.position[2]*0.0112 + 0.0114

        #     # print('Posição 1:', self.position.T)

        # if self.position2 is not None:
        #     #Corrected Position Camera 2
            
        #     self.position2[0] = self.position2[0]*0.946 + 0.00334
        #     self.position2[1] = self.position2[1]*0.892 - 0.656
        #     self.position2[2] = self.position2[2]*0.815 - 1.55

        #     # Z correction
        #     self.position2[0] -= -self.position2[2]*0.00821 + 0.00828
        #     self.position2[1] -= self.position2[2]*0.025 - 0.0366

        #     # print('Posição 2:', self.position2.T)
                

        # # Estimation using 2 cameras
        # if self.position is not None and self.position2 is not None:
            
        #     #Verify if the camera 1 was in oclusion
        #     if self.count_cam1_off:
                
        #         pos_error = abs(self.position - self.position2)

        #         #Verify if the error beetween the camera 1 and camera 2 is too high. As the camera 1 return from oclusion, 
        #         #it measurements are not reliable.
        #         if pos_error[0] > 0.05 or pos_error[1] > 0.05 or pos_error[2] > 0.1:
    
        #             self.position_mean = self.position2
        #         #In case the error is small, the camera 1 is considered again
        #         else:
                    
        #             self.position_mean = 0.7*self.position + 0.3*self.position2
        #             self.count_cam1_off = False

        #     #Verify if the camera 2 was in oclusion
        #     elif self.count_cam2_off:

        #         pos_error = abs(self.position - self.position2)
                
        #         #Verify if the error beetween the camera 1 and camera 2 is too high. As the camera 2 return from oclusion, 
        #         #it measurements are not reliable.
        #         if pos_error[0] > 0.05 or pos_error[1] > 0.05 or pos_error[2] > 0.1:

        #             self.position_mean = self.position

        #         #In case the error is small, the camera 1 is considered again
        #         else:
                    
        #             self.position_mean = 0.7*self.position + 0.3*self.position2
        #             self.count_cam2_off = False

        #     else:
                
        #         if self.pos_ant is not None and self.pos2_ant is not None:

        #             pos_1_error = abs(self.position - self.pos_ant)
        #             pos_2_error = abs(self.position2 - self.pos2_ant)
        #             # print('Erro pos 1: {0} \n Erro pos 2: {1}'.format(pos_1_error, pos_2_error))
                    
        #             #Verify if there is no spike in measurements, often caused by oclusions or distortions in detection
        #             #In case the difference is greater than 0.2 the measurements of camera 1 is not considered
        #             if pos_1_error[0] > 0.1 or pos_1_error[1] > 0.1 or pos_1_error[2] > 0.1:
                        
        #                 self.position_mean = self.position2
        #                 self.count_cam1_off = True


        #             #Verify if there is no spike in measurements, often caused by oclusions or distortions in detection
        #             #In case the difference is greater than 0.2 the measurements of camera 2 is not considered
        #             elif pos_2_error[0] > 0.1 or pos_2_error[1] > 0.1 or pos_2_error[2] > 0.1:
                        
        #                 self.position_mean = self.position
        #                 self.count_cam2_off = True

        #             else:
        #                 self.position_mean = 0.7*self.position + 0.3*self.position2

            

        #     print('Duas câmeras')

                
        # if self.position is not None and self.position2 is not None:

        #     self.position_mean = 0.7*self.position + 0.3*self.position2                    

        # #Estimation using only camera 1
        # if self.position is not None and self.position2 is None:

        #     self.position_mean = self.position

        #     self.count_cam2_off = True
        #     # print('Câmera 1 Apenas')

        # #Estimation using only camera 2
        # if self.position is None and self.position2 is not None:

        #     self.position_mean = self.position2
        #     self.count_cam1_off = True
        #     # print('Câmera 2 Apenas')


        # self.pos_ant = self.position
        # self.pos2_ant = self.position2


        # if self.cam_vec_off is False and self.cam_vec is not None:

        #     self.cx_list.append(self.cam_vec[0])
        #     self.cy_list.append(self.cam_vec[1])
        #     self.cz_list.append(self.cam_vec[2])

        #     mean_cx = np.mean(self.cx_list)
        #     mean_cy = np.mean(self.cy_list)
        #     mean_cz = np.mean(self.cz_list)

        #     self.cam_vec_mean = np.array([[mean_cx, mean_cy, mean_cz]]).T
                    

        # if self.cam_vec2_off is False and self.cam_vec2 is not None:

        #     self.cx2_list.append(self.cam_vec2[0])
        #     self.cy2_list.append(self.cam_vec2[1])
        #     self.cz2_list.append(self.cam_vec2[2])

        #     mean_cx2 = np.mean(self.cx2_list)
        #     mean_cy2 = np.mean(self.cy2_list)
        #     mean_cz2 = np.mean(self.cz2_list)

        #     self.cam_vec2_mean = np.array([[mean_cx2, mean_cy2, mean_cz2]]).T

                    
                
                

        cv.imshow('Drone Camera', self.image)
        # cv.imshow('Drone Camera 2', image2)
        cv.waitKey(1)