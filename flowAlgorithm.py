import numpy as np


q = 9  
omega = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])  # Weights for velocity vectors
epsilon = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])  # Velocity vectors

# Define the flow algorithm function
def flow_algorithm(rho, N, omega_fr, omega_tilde, nbFrames):
    velocity_field = np.zeros((nbFrames, rho.shape[1], rho.shape[2],2))  # Initialize velocity field
    f_s = np.zeros((q, rho.shape[1], rho.shape[2], nbFrames))  # Initialize distribution function
    rho_n = np.zeros((rho.shape[1], rho.shape[2],nbFrames ))
    for t in range(nbFrames-1):
        for x in range(1,rho.shape[1]):
            for y in range(rho.shape[2]):
                for i in range(q):
                    f_s[i,x,y,t] = omega[i] * rho[t,x,y]  # Initialize probabilities

        for n in range(N):
            for x in range(rho.shape[1]):
                for y in range(rho.shape[2]): 
                    for i in range(q):
                        rho_xy_t = rho[t,x, y]
                        v_n_xy_t = velocity_field[t, x, y]
                        a = np.dot(epsilon[i], v_n_xy_t)
                        norm_v = np.linalg.norm(v_n_xy_t)**2
                        sum_v_1 = a**2
                        # except RuntimeWarning:
                            # print(sum_v_1)
                        sum_v_2 = 1.5*norm_v
                        sum_v = v_n_xy_t + (9/2.0)*sum_v_1-sum_v_2
                        f_eq =   1 + 3*np.dot(epsilon[i], sum_v )  # 
                        f_eq = omega[i] * f_eq
                        f_eq =  rho_xy_t * f_eq
                        #Calculate equilibrium distribution
                        f_c = f_s[i,x,y,t] - omega_fr*(f_s[i,x,y,t]-f_eq) # Collision

                        new_x = x + epsilon[i][0]
                        new_y = y + epsilon[i][1]
                
                        # Vérifier que les indices restent dans les limites du tableau
                        if 0 <= new_x < rho.shape[1] and 0 <= new_y < rho.shape[2]:
                            f_s[i,new_x,new_y,t+1] = f_c  # Stream
            for x in range(rho.shape[1]):
                for y in range(rho.shape[2]):
                    rho_n[x,y,t+1] = 0
                    for i in range(q):                              
                        rho_n[x,y,t+1] += f_s[i,x,y,t+1] 
            for x in range(rho.shape[1]):
                for y in range(rho.shape[2]): 
                    for i in range(q):
                        new_x = x + epsilon[i][0]
                        new_y = y + epsilon[i][1]
                
                        # Vérifier que les indices restent dans les limites du tableau
                        if 0 <= new_x < rho.shape[1] and 0 <= new_y < rho.shape[2]:
                            f_s[i,x,y,t] = f_s[i,x,y,t]-omega_tilde*(rho_n[new_x,new_y,t+1]-rho[t+1,new_x,new_y]) # Correction
                    if rho[t,x, y]!=0:
                            velocity_field[t, x, y] = calculate_velocity(f_s[:,x,y,t], rho[t,x, y])  # Calculate velocity field
                    #np.where(rho[x, y, t]!=0,velocity_field[t, x, y],calculate_velocity(f_s[:,x,y,t], rho[x, y, t]))
                        

    return velocity_field

def calculate_velocity(f_s_xyt, rho_xy):
    velocity = np.zeros(2)
    for i in range(q):
        velocity += epsilon[i] * f_s_xyt[i]
    return velocity / rho_xy



