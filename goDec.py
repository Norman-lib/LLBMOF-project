from scipy.linalg import qr, fractional_matrix_power
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error
import numpy as np



def GoDec(video,r,k,epsilon,q):
    f,x,y=video.shape
    X_reshaped=video.reshape((f,x*y)).T
    m,n = X_reshaped.shape
    St=np.zeros(X_reshaped.shape)
    Lt=np.zeros(X_reshaped.shape)
    error_history = []
    t=0
    while np.linalg.norm(X_reshaped-Lt-St)/np.linalg.norm(X_reshaped)>epsilon and t<20:
        # if (t+1)%100==0:
        #     print("iteration hh",t, "Error ",error_history[-1] if len(error_history)>0 else "None")
        t += 1
        Ltilde = np.dot(np.linalg.matrix_power(np.dot((X_reshaped - St), (X_reshaped - St).T), q),(X_reshaped - St))
        A1 = np.random.randn(n, r)
        Y1 = Ltilde.dot(A1)
        A2=Y1
        Y2 = (Ltilde.T).dot( Y1)
        Q2, R2 = qr(Y2, mode='economic')
        Y1 = Ltilde.dot(Y2)
        Q1, R1 = qr(Y1, mode='economic')

        # A1, A2 = qr(Y1, mode='economic')
        # #if q == 0:
        #    Lt = np.dot(Y1, np.dot(pinv(np.dot(A2.T,Y1)), Y2.T))
        #else:
        if np.linalg.matrix_rank((A2.T).dot( Y1)) < r:
            r = np.linalg.matrix_rank(np.dot(A2.T, Y1))
            continue

        # Lt = np.dot(Q1, np.dot(pinv(R1), R2.T))
        Lt = fractional_matrix_power(R1.dot(pinv(np.dot(A2.T,Y1))).dot(R2.T), (1/(2*q+1)))
        Lt = Q1.dot(Lt).dot(Q2.T)
            
        S_flat = np.abs(X_reshaped - Lt).flatten()
        
        # Keep only the k largest elements of S
        k_largest_indices = np.argsort(S_flat)[0:len(S_flat)-k]
        S_flat[k_largest_indices] = 0
        St = S_flat.reshape(Lt.shape)
        
        error_history.append(mean_squared_error(X_reshaped, Lt+St))
    Lt = Lt.T.reshape(f,x,y)
    St = St.T.reshape(f,x,y)
    Wt = video - Lt - St
    return Lt, St, Wt, error_history
            