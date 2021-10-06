import numpy as np
from numpy.core.numeric import NaN
from quaternion_euler_utility import deriv_quat
from quaternion_euler_utility import quat_prod
from scipy.linalg import solve_continuous_are as solve_lqr


class Controller:

    #Constants
    J = np.array([[16.83*10**-3, 0, 0],
                  [0, 16.83*10**-3, 0],
                  [0, 0, 28.34*10**-3]])
    Ixx = 16.83*10**-3
    Iyy = 16.83*10**-3
    Izz = 28.34*10**-3
    #Mass and Gravity
    M, G = 1.03, 9.82

    KT = 1.435*10**-5
    KD = 2.4086*10**-7
    L = 0.26


    
    def __init__(self, inner_length):
        
        self.il = inner_length

        self.nd_ant = 1
        self.ed1_ant = 0
        self.ed2_ant = 0
        self.ed3_ant = 0

        self.ang_ant_des = np.zeros((3,1))
    

    def f2w(self,f,m):
        """""
        Translates F (Thrust) and M (Body x, y and z moments) into eletric motor angular velocity (rad/s)
        input:
            f - thrust 
            m - body momentum in np.array([[mx, my, mz]]).T
        outputs:
            F - Proppeler Thrust - engine 1 to 4
            w - Proppeler angular velocity - engine 1 to 4
            F_new - clipped thrust (if control surpasses engine maximum)
            M_new - clipped momentum (same as above)
        """""
        x = np.array([[self.KT, self.KT, self.KT, self.KT],
                      [-self.L*self.KT, 0, self.L*self.KT, 0],
                      [0, -self.L*self.KT, 0, self.L*self.KT],
                      [-self.KD, self.KD, -self.KD, self.KD]])

        y = np.array([float(f), float(m[0]), float(m[1]), float(m[2])]).T
        
        u = np.linalg.solve(x, y)

        
        modules = np.zeros(4)
        for k in range(4):
            modules[k] = -1 if u[k] < 0 else 1
        w_1 = np.sqrt(np.abs(u[0]))*modules[0]
        w_2 = np.sqrt(np.abs(u[1]))*modules[1]
        w_3 = np.sqrt(np.abs(u[2]))*modules[2]
        w_4 = np.sqrt(np.abs(u[3]))*modules[3]
            
        w = np.array([[w_1,w_2,w_3,w_4]]).T

        FM_new = np.dot(x, u)
        
        F_new = FM_new[0]
        M_new = FM_new[1:4]
        
        # step_effort = (u*K_F/(T2WR*M*G/4)*2)-1
        
        return w, F_new, M_new
        
    ######################################## PID CONTROL APPROACH #############################################################    

    def att_control_PD(self, ang_atual, ang_vel_atual, ang_des):
        
        phi = float(ang_atual[0])
        theta = float(ang_atual[1])
        psi = float(ang_atual[2])

        #PID gains Real States
        Kp = np.array([[30, 0 ,0],
                       [0, 30, 0],
                       [0, 0, -0.4]])*2.45
        Kd = np.array([[8, 0, 0],
                       [0, 8, 0],
                       [0, 0, -1]])*0.8


        #PD gains Estimated States
        # Kp = np.array([[200, 0 ,0],
        #                [0, 200, 0],
        #                [0, 0, 120]])*1.5
        # Kd = np.array([[50, 0, 0],
        #                [0, 50, 0],
        #                [0, 0, 45]])*1

        # Kp = np.array([[20, 0 ,0],
        #                [0, 20, 0],
        #                [0, 0, 5]])*0.6
        # Kd = np.array([[12, 0, 0],
        #                [0, 12, 0],
        #                [0, 0, 1]])*1

        # Kp = np.array([[250, 0 ,0],
        #                [0, 250, 0],
        #                [0, 0, 10]])*5.4
        # Kd = np.array([[40, 0, 0],
        #                [0, 40, 0],
        #                [0, 0, 35]])*1.3
        
        angle_error = ang_des - ang_atual

        ang_vel_des = (ang_des - self.ang_ant_des)/0.01

        ang_vel_error = ang_vel_des - ang_vel_atual

        #Compute Optimal Control Law

        print(angle_error.T)
        # print(ang_vel_error.T)

        T = np.array([[1/self.Ixx, np.sin(phi)*np.tan(theta)/self.Iyy, np.cos(phi)*np.tan(theta)/self.Izz],
                      [0, np.cos(phi)/self.Iyy, -np.sin(phi)/self.Izz],
                      [0, np.sin(phi)/np.cos(theta)/self.Iyy, np.cos(phi)/np.cos(theta)/self.Izz]])

        # print(Kp@angle_error)

        u = np.linalg.inv(T)@(Kp@angle_error + Kd@ang_vel_error)
        # u = (Kp@angle_error + Kd@ang_vel_error)

        #Optimal input
        tau_x = float(u[0])
        tau_y = float(u[1])
        tau_z = float(u[2])

        self.ang_ant_des = ang_des

        return tau_x, tau_y, tau_z
    
    def pos_control_PD(self, pos_atual, pos_des, vel_atual, vel_des, accel_des, psi):

        #PD gains Real States
        Kp = np.array([[2, 0 ,0],
                       [0, 2, 0],
                       [0, 0, 8]])*2
        Kd = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 3]])*0.5

        # Kp = np.array([[2, 0 ,0],
        #                [0, 2, 0],
        #                [0, 0, 1.5]])*1.5
        # Kd = np.array([[3, 0, 0],
        #                [0, 2, 0],
        #                [0, 0, 1]])*1.5

        dpos_error = pos_des - pos_atual

        vel_error = vel_des - vel_atual

        n = accel_des/np.linalg.norm(accel_des)
        t = vel_des/np.linalg.norm(vel_des)
        b = np.cross(t, n, axis=0)

        if np.isnan(b).any:
            pos_error = dpos_error
        else:
            pos_error = (dpos_error.T@n)@n + (dpos_error.T@b)@b


        # u = Kp@dpos_error + Kd@vel_error

        # theta_des = np.arctan2(u[0], (u[2]+9.82))

        # phi_des = np.arctan2(-u[1], (u[2]+9.82)*np.cos(theta_des))

        # T = 1.03*(u[2] + 9.82)/(np.cos(theta_des)*np.cos(phi_des))

        
        # alo = 1
        rddot_c = accel_des + Kd@vel_error + Kp@pos_error

        T = self.M*(self.G + rddot_c[2])

        phi_des = (rddot_c[0]*np.sin(psi) - rddot_c[1]*np.cos(psi))/self.G
        theta_des = (rddot_c[0]*np.cos(psi) + rddot_c[1]*np.sin(psi))/self.G

        return T, phi_des, theta_des



    #################################### TRAJECTORY PLANNER FUNCTIONS ######################################################
    
    #Returns the derivatives
    def polyT(self, n, k, t):

        T = np.zeros((n,1))
        D = np.zeros((n,1))

        for i in range(1, n+1):
            D[i-1] = i - 1
            T[i-1] = 1

        for j in range(1, k+1):
            for i in range(1, n+1):
                T[i-1] = T[i-1]*D[i-1]

                if D[i-1]>0:
                    D[i-1] = D[i-1] - 1
                

        for i in range(1, n+1):
            T[i-1] = T[i-1]*t**D[i-1]
        
        T = T.T

        return T

    #Get the optimal snap functions coefficients 
    def getCoeff_snap(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((8*n, 8*n))
        b = np.zeros((8*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 4):
            A[row, 0:8] = self.polyT(8, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 8*(i+shift):8*(i+1+shift)] = self.polyT(8, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 7):
                    A[row, 8*(j-1):8*(j)] = self.polyT(8, k, t[j])
                    A[row, 8*(j):8*(j+1)] = -self.polyT(8, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 8)

        return A, b, c_matrix

    #Compute the snap trajectory equations at time 't'
    def equation_snap(self, t, c_matrix, eq_n):
        x = self.polyT(8, 0, t)
        v = self.polyT(8, 1, t)
        a = self.polyT(8, 2, t)
        j = self.polyT(8, 3, t)
        s = self.polyT(8, 4, t)
        

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        J = np.sum(j*c_matrix[eq_n,:])
        S = np.sum(s*c_matrix[eq_n,:])
        

        return P, V, A, J, S
    
    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_snap(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []
        j_list = []
        s_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)
        
        return x_list, v_list, a_list, j_list, s_list


    #Get the optimal acceleration functions coefficients 
    def getCoeff_accel(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((4*n, 4*n))
        b = np.zeros((4*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 2):
            A[row, 0:4] = self.polyT(4, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 4*(i+shift):4*(i+1+shift)] = self.polyT(4, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 3):
                    A[row, 4*(j-1):4*(j)] = self.polyT(4, k, t[j])
                    A[row, 4*(j):4*(j+1)] = -self.polyT(4, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 4)

        return A, b, c_matrix

    #Compute the acceleration trajectory equations at time 't'
    def equation_accel(self, t, c_matrix, eq_n):
        x = self.polyT(4, 0, t)
        v = self.polyT(4, 1, t)
        a = self.polyT(4, 2, t)

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        

        return P, V, A

    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_accel(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
        
        return x_list, v_list, a_list
