from tkinter import W
import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    r1, c1 = It.shape
    r2, c2 = It.shape
    
    splinet  = RectBivariateSpline(np.linspace(0, r1, r1), np.linspace(0, c1, c1), It)  
    splinet1 = RectBivariateSpline(np.linspace(0, r2, r2), np.linspace(0, c2, c2), It1)
    
    Iy, Ix = np.gradient(It1) # Affine subtraction
    spline_x = RectBivariateSpline(np.linspace(0, r2, r2), np.linspace(0, c2, c2), Ix)
    spline_y = RectBivariateSpline(np.linspace(0, r2, r2), np.linspace(0, c2, c2), Iy)
    
    M = np.eye(3)
    
    #coordinates for the template image
    x, y = np.mgrid[0:c1, 0:r1]
    
    x_c = np.reshape(x, (1, -1))
    y_c = np.reshape(y, (1, -1))
    
    #[x, y, 1]
    coor = np.vstack((x_c, y_c, np.ones((1, r1*c1))))
    p  = np.zeros(6)
    dp = np.ones(6) #six parameters to be determined
    n=1
    
    while(np.square(dp).sum()>threshold and n<num_iters):
        M=np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0, 0, 1]])
        warp = M@coor #3*N  
        
        #xp and yp coordinates
        warp_x = warp[0]
        warp_y = warp[1]
        
        #gradient splines
        grad_x = spline_x.ev(warp_y, warp_x).flatten()
        grad_y = spline_y.ev(warp_y, warp_x).flatten()
        
        warp_final = splinet1.ev(warp_y,warp_x).flatten()
        T = splinet.ev(y, x).flatten()
        
        #error image
        error = np.reshape(T-warp_final, (len(warp_x), 1))
        
        A1=np.multiply(grad_x, x_c)
        A2=np.multiply(grad_x, y_c)
        A3=np.reshape(grad_x, (1,-1))
        A4=np.multiply(grad_y, x_c)
        A5=np.multiply(grad_y, y_c)
        A6=np.reshape(grad_y, (1,-1))
        A = np.vstack((A1, A2, A3, A4, A5, A6)) #this is the Jaconian and the gradient of I
        A=A.T
                
        H = A.T@A#We calculate the Hessian
        
        dp = np.linalg.inv(H) @ A.T @ error 
        p = (p + dp.T).ravel()
        n+=1
        
    M = np.array([[1+p[0], p[1],p[2]], [p[3], 1+p[4], p[5]], [0, 0, 1]])
    return M