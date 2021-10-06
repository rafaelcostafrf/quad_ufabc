#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as rot

from quad_ros import quad_robot
from quad_control import Controller
from computer_vision import ComputerVision
import matplotlib.pyplot as plt
import pickle

################################################################################################################################



rospy.init_node('Test')
quad_ufabc = quad_robot('quad')
controller = Controller(1)
comp_vis = ComputerVision()
quad_ufabc.reset()
quad_ufabc.step([0, 0, 0, 0])

#Trajectory Planner

infile = open('trajectory.p', 'rb')
traj = pickle.load(infile)
infile.close()

cont = 0

# fig, x = plt.subplots(1, 1, figsize=(9,7))
# x.plot(traj['z'])
# x.plot(traj['x'])
# plt.show()


while True:

    if comp_vis.image is not None:
        # print(comp_vis.image)
        comp_vis.img_show()

    if cont < len(traj['x']):
        #Translation real states
        pos_real = np.asarray(quad_ufabc.position).reshape(3,1)
        vel_real = np.asarray(quad_ufabc.velocity).reshape(3,1)

        #Position Control
        pos_ref = np.array([[traj['x'][cont], traj['y'][cont], traj['z'][cont]]]).T
        vel_ref = np.array([[traj['dx'][cont], traj['dy'][cont], traj['dz'][cont]]]).T
        accel_ref = np.array([[traj['ddx'][cont], traj['ddy'][cont], traj['ddz'][cont]]]).T

        psi = traj['psi'][cont]

        T, phi_des, theta_des = controller.pos_control_PD(pos_real, pos_ref, vel_real, vel_ref, accel_ref, psi)
        
        ang_des = np.array([[float(phi_des), float(theta_des), float(psi)]]).T
        # ang_des = np.zeros((3,1))
        #Inner Loop - Attitude Control

        #Get Euler Angles 

        ang_real = np.asarray(quad_ufabc.attitude_euler).reshape(3,1)
        # print('angle', ang_real.T)
        
        #Get Angular Velocities    

        ang_vel_real = np.asarray(quad_ufabc.angular_vel).reshape(3,1)

        #Run Attitude Control
        taux, tauy, tauz = controller.att_control_PD(ang_real, ang_vel_real, ang_des)

        w, _, _ = controller.f2w(T, [taux, tauy, tauz])

        w[0] = -w[0]
        w[2] = -w[2]

        quad_ufabc.step(w/10)
    
    else:

        quad_ufabc.step([-20, 20, -20, 20])

    cont += 1

    quad_ufabc.rate.sleep()