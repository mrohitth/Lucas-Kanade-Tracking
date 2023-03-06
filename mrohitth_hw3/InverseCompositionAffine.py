import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
    p  = np.zeros(6)
    dp = np.ones(6) #six parameters to be determined
    M = np.eye(3)
    r1, c1 = It.shape
    r2, c2 = It.shape

    splinet  = RectBivariateSpline(np.linspace(0, r1, r1), np.linspace(0, c1, c1), It) 
    splinet1 = RectBivariateSpline(np.linspace(0, r2, r2), np.linspace(0, c2, c2), It1)
    
    Iy, Ix = np.gradient(It)    #Affine subtraction
    spline_x = RectBivariateSpline(np.linspace(0, r1, r1),np.linspace(0, c1, c1), Ix)
    spline_y = RectBivariateSpline(np.linspace(0, r1, r1),np.linspace(0, c1, c1), Iy)
    
    x, y = np.mgrid[0:c1, 0:r1]
    x_c = np.reshape(x, (1, -1))
    y_c = np.reshape(y, (1, -1))
    
    #[x, y, 1]
    coor = np.vstack((x_c, y_c, np.ones((1, r1*c1))))
    
    grad_x = spline_x.ev(y, x).flatten()
    grad_y = spline_y.ev(y, x).flatten()
    
    T = splinet.ev(y, x).flatten()
   
    A1 = np.multiply(grad_x, x_c)
    A2 = np.multiply(grad_x, y_c)
    A3 = np.reshape(grad_x, (1, -1))
    A4 = np.multiply(grad_y, x_c)
    A5 = np.multiply(grad_y, y_c)
    A6 = np.reshape(grad_y, (1, -1))
    A = np.vstack((A1, A2, A3, A4, A5, A6)) #Jaconian and the gradient of I
    A = A.T
    H = A.T@A #Hessian
    n = 1
    
    while(np.square(dp).sum()>threshold and n<num_iters):        
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0, 0, 1]])
        warp = M@coor        

        #xp and yp coordinates
        warp_x = warp[0]
        warp_y = warp[1]
        
        # gradient splines
        warp_final = splinet1.ev(warp_y, warp_x).flatten()
        
        #error image
        error = np.reshape(T-warp_final, (len(warp_x), 1))
        
        dp = np.linalg.inv(H) @ A.T @ error 
        p = (p + dp.T).ravel()
        n+=1
        
        dM = np.vstack((dp.reshape(2, 3), [0, 0, 1]))
        M = M @ np.linalg.inv(dM)
        
    return M
