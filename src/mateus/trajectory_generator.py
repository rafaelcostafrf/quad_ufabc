import numpy as np
import pickle
from quad_control import Controller
import matplotlib.pyplot as plt


controller = Controller(1)

x_wp = np.array([[0.7, 0.7, 0.5, 0.25, 0]]).T
y_wp = np.array([[0, 0.5, 1, 0.5, 0]]).T
z_wp = np.array([[1, 1.25, 2.5, 3.75, 2]]).T
psi_wp = np.array([[0, 0, 0, 0, 0]]).T

time = np.array([0, 5, 10, 15, 20])*5

step_controller = 0.01

_, _, x_matrix = controller.getCoeff_snap(x_wp, time)
_, _, y_matrix = controller.getCoeff_snap(y_wp, time)
_, _, z_matrix = controller.getCoeff_snap(z_wp, time)
_, _, psi_matrix = controller.getCoeff_accel(psi_wp, time)

x_ref, dotx_ref, ddotx_ref, _, _ = controller.evaluate_equations_snap(time, step_controller, x_matrix)
y_ref, doty_ref, ddoty_ref, _, _ = controller.evaluate_equations_snap(time, step_controller, y_matrix)
z_ref, dotz_ref, ddotz_ref, _, _ = controller.evaluate_equations_snap(time, step_controller, z_matrix)
psi_ref, _, _ = controller.evaluate_equations_accel(time, step_controller, psi_matrix)

#test

# t = np.arange(0, 0.01, 20)

fig, x = plt.subplots(1,1,figsize=(9,7))
x.plot(z_ref)
x.plot(y_ref)
x.plot(x_ref)
plt.show()

trajectory_1 = {'x': x_ref, 'y': y_ref, 'z': z_ref, 'dx': dotx_ref, 'dy': doty_ref, 'dz': dotz_ref, 'ddx': ddotx_ref,
                'ddy': ddoty_ref, 'ddz': ddotz_ref, 'psi': psi_ref}

outfile = open('trajectory.p', 'wb')

pickle.dump(trajectory_1, outfile)
outfile.close()
