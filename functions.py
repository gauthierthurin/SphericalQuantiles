# Python 3.10.8

import pyshtools as pysh  
#https://nbviewer.org/github/SHTOOLS/SHTOOLS/blob/master/examples/notebooks/grids-and-coefficients.ipynb
import numpy as np 
import matplotlib.pyplot as plt 
import sphericart as sc 
from time import time 
import ot

from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.visualization as visualization

#############################################################################
# Basic functions 
#############################################################################

def cartesian2polar(xyz):
    '''
    returns (phi,theta) = (longitude,latitude) where
    - phi in [0,2pi]
    - theta in [-pi/2,pi/2]
    '''
    phi = np.arctan2(xyz[1],xyz[0]) #in [-pi,pi]
    theta = np.arccos(xyz[2]) # in [0,pi]
    # This gives spherical coordinates with a different parameterization, 
    # see https://fr.wikipedia.org/wiki/Coordonnées_sphériques.
    # To get the desired angles, it suffices to translate as follows.
    theta = np.pi/2 - theta  #in [-pi/2,pi/2]
    phi = phi + 2*np.pi*(1*(phi<0)) # in [0,2pi]
    return([phi,theta])

def polar2cartesian(phi,theta):
    '''
    phi : longitude [0,2pi]
    theta : latitude [-pi/2,pi/2]
    '''
    #phi = phi - np.pi # in [-pi,pi]
    phi = phi - 2*np.pi*(1*(phi>np.pi)) # in [-pi,pi]
    theta = np.pi/2 - theta # in [0,pi]
    xyz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])
    return(xyz)

def func_Frechetmean(data):
    '''Frechet mean computed with geomstats'''
    sphere = Hypersphere(dim=2)
    mean = FrechetMean(sphere)
    mean.fit(data)
    return(mean.estimate_)

def func_GeomMedian(data):
    '''Frechet mean computed with geomstats'''
    sphere = Hypersphere(dim=2)
    med = GeometricMedian(sphere)
    med.fit(data)
    return(med.estimate_)

def def_ax_sphere3D(el,az,sph_colour="ghostwhite",sph_alpha=0.5):
    ax = plt.axes(projection='3d')
    ax.view_init(el, az)
    ax.set_xlim(-0.65,0.65)
    ax.set_ylim(-0.65,0.65)
    ax.set_zlim(-0.65,0.65)
    ax.set_axis_off()

    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color=sph_colour,alpha=sph_alpha)
    return( ax )

def plot_3d_scatter(data,ax,colour='red',sz=5, alpha = 1,marker='o'):
    # Check that data is 3D (data should be Nx3)
    d = np.shape(data)[1]
    if d != 3:
        raise Exception("data should be of shape Nx3, i.e., each data point should be 3D.")
    ax.scatter(data[:,0],data[:,1],data[:,2],s=sz,c=colour,alpha=alpha,marker=marker)
    return ax

def contour_unif(tau,size=100):
    ''' 
    Returns a set of point with a given latitude depending on tau ; this corresponds to a reference quantile contour (wrt to the uniform distribution) of order tau, oriented on the North Pole
    '''
    LON = np.linspace(0,2*np.pi,size)
    #LAT = np.pi/2 - tau*np.pi # between -pi/2 and pi/2 
    LAT = np.arccos(1 - 2*tau)
    LAT = np.pi/2 - LAT
    phi,tht = np.meshgrid(LON,LAT) 
    grid_3D = polar2cartesian(phi,tht)
    grid_3D = grid_3D.reshape((3,len(LON))).T
    return(grid_3D)

def signcurve_unif(LON,size=100):
    ''' LON is any longitude between 0 and 2pi. Returns a set of points with a given longitude ; This corresponds to reference sign curves with respect to the North Pole '''
    LAT = np.linspace(0,0.99,size)
    LAT = np.pi/2 - LAT*np.pi # between -pi/2 and pi/2 
    phi,tht = np.meshgrid(LON,LAT)
    grid_3D = polar2cartesian(phi,tht)
    grid_3D = grid_3D.reshape((3,size)).T
    return(grid_3D)

def RodriguesRot(v,k,angle):
    '''
    Rodrigues rotation formula.
    - v is the vector to be rotated
    - k is a unit vector giving an axis of rotation
    - the function rotates v around k by an angle 'angle', in radians, according to the right-hand-rule.
    '''
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    v_rot = v + np.sin(angle) * (K @ v) + (1 - np.cos(angle))*(K@K@v)
    return(v_rot)

def rotateData(data,MeanData):
    ''' 
    Rotate data oriented wrt to (0,0,1) to data oriented wrt MeanData 
    '''
    k = MeanData-[0,0,1]
    k = k/np.linalg.norm(k)

    Newdata = []
    for x in data:
        Newdata.append( RodriguesRot(x,-k,np.pi) )
    Newdata = np.array(Newdata)
    return(Newdata)

#############################################################################
# Functions to solve the continuous OT problem via spherical harmonics coefficients
#############################################################################

def weights_W(coeffs,s=1):
    '''
    - returns the matrix of weight W 
    - Result of same shape as `coeffs`.
    '''
    LMAX = np.shape(coeffs.to_array())[1]
    vec = np.arange(LMAX)
    L,M = np.meshgrid(vec,vec)
    W = (L**2+M**2)**s
    # take care of non dividing by zero
    W[0,0] = 1 
    W = 1/W
    return(W)

def c_transform_fft2(u, C, eps):
    """Calculate the c_transform of u"""
    arg = (u.to_array() - C )/eps
    M = np.max(arg)
    to_sum = np.exp(arg - M)
    theta = u.lats()*np.pi/180
    theta = np.pi/2 - theta
    argD = np.mean(to_sum * np.sin(theta).reshape(len(theta),1))
    return(-eps*np.log(argD) -eps*M )

def grad_heps_fft2(u, C, eps):
    """
    Calculate the gradient of h_eps
    """
    arg = (u.to_array() - C )/eps
    M = np.max(arg) 
    F_u = np.exp(arg - M)
    theta = u.lats()*np.pi/180 
    theta = np.pi/2 - theta
    argD = np.mean(F_u * np.sin(theta).reshape(len(theta),1))
    F_u = pysh.shclasses.SHGrid.from_array(F_u/argD)
    return(F_u.expand())

def h_eps_fft2(u, C, eps):
    """
    Calculate the function h_eps whose expectation equals H_eps.
    """
    return(c_transform_fft2(u, C, eps)-eps)

def cost_fft2(grid,y):
    '''
    - Quadratic cost between any y and a uniform grid with format `grid` from pySHTOOLS.
    - y must be in cartesian coordinates, (xyz)
    - Result of same shape as `grid`.
    '''
    theta = grid.lats()*np.pi/180 # in [-pi/2,pi/2]
    phi = grid.lons()*np.pi/180 # in [0,2pi]
    theta = np.pi/2 - theta  # in [0,pi]
    phi = phi - 2*np.pi*(1*(phi>np.pi)) # in [-pi,pi]
    LON,LAT = np.meshgrid(phi,theta)
    scal_prod_matrix = np.cos(LON)*np.sin(LAT)*y[0] + np.sin(LON)*np.sin(LAT)*y[1] + np.cos(LAT)*y[2]
    #xyz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])
    # https://fr.wikipedia.org/wiki/Coordonnées_sphériques.
    #scal_prod_matrix = np.minimum(1, scal_prod_matrix)
    #scal_prod_matrix = np.maximum(-1, scal_prod_matrix)
    C = 0.5*np.arccos(scal_prod_matrix)**2
    return(C)

def Robbins_Monro_Algo(Y, eps=0.1, gamma= 1, c = 3/4, epoch = 1,l_max=30): 
    '''
    - grid_x = a SHGrid object. It is the initialisation for the function u(x).
    - Y = sample from nu, in cartesian coordinates, of size (n,3) for n=number of samples, in dimension 3.  
    '''
    n = Y.shape[0]
    n_iter = n*epoch
    # Tirage des Y le long des itérations
    sample = np.random.choice(a=np.arange(n), size=n_iter)
    
    # Stockage des estimateurs recursifs.
    W_hat_storage = np.zeros(n_iter)
    h_eps_storage = np.zeros(n_iter)

    # Initialisation du vecteur u
    fft_u = np.zeros((2, l_max+1, l_max+1))
    fft_u = pysh.shclasses.SHCoeffs.from_array(fft_u)
    u = fft_u.expand()
    
    # Choix de la matrice de poids W pour l'algorithme basé sur la FFT. 
    W = weights_W(fft_u)
    
    # Premiere iteration pour lancer la boucle. 
    y_0 = Y[sample[0],:]
    C = cost_fft2(u, y_0)
    heps = h_eps_fft2(u, C, eps)
    W_hat_storage[0] = heps
    h_eps_storage[0] = heps

    # Boucle de Robbins Monro.
    for k in range(1,n_iter):
        # Tirage d'une réalisation selon la loi mu.
        y = Y[sample[k],:]
        C = cost_fft2(u, y)
        
        # Mise à jour de la valeur de fft_u 
        grad = - gamma/((k+1)**c) * W *  grad_heps_fft2(u, C, eps).to_array()
        fft_u = fft_u + pysh.shclasses.SHCoeffs.from_array(grad)  
        # Mise à zéro de son intégrale
        u = fft_u.expand()
        theta = np.pi/2 - u.lats()*np.pi/180 
        u.data = u.data - np.mean(u.data * np.sin(theta).reshape(len(theta),1))
        #fft_u = fft_u.to_array()
        #fft_u[:, 0, 0] = 0
        #fft_u = pysh.shclasses.SHCoeffs.from_array(fft_u,csphase = 1)
        fft_u = u.expand()

        # Stockage de la valeur de h_eps 
        h_eps_storage[k] = h_eps_fft2(u, C, eps)
    
        # Evaluation de l'approximation de la divergence de Sinkhorn.
        W_hat_storage[k] = k/(k+1) * W_hat_storage[k-1] + 1/(k+1) * h_eps_storage[k]
        
        
    L = [u, W_hat_storage]
    
    return(L)
    
    


def Robbins_Monro_Algo_faster(Y, eps=0.1, gamma= 1, c = 3/4, epoch = 1,l_max=30):
    '''
    - grid_x = a SHGrid object. It is the initialisation for the function u(x).
    - Y = sample from nu, in cartesian coordinates, of size (n,3) for n=number of samples, in dimension 3.  
    '''
    n = Y.shape[0]
    n_iter = n*epoch
    # Tirage des Y le long des itérations
    sample = np.random.choice(a=np.arange(n), size=n_iter)

    # Initialisation du vecteur u
    fft_u = np.zeros((2, l_max+1, l_max+1))
    fft_u = pysh.shclasses.SHCoeffs.from_array(fft_u)
    u = fft_u.expand()

    # Choix de la matrice de poids W pour l'algorithme basé sur la FFT. 
    W = weights_W(fft_u)

    # Premiere iteration pour lancer la boucle. 
    y_0 = Y[sample[0],:]
    C = cost_fft2(u, y_0)

    # Boucle de Robbins Monro.
    for k in range(1,n_iter):
        # Tirage d'une réalisation selon la loi mu
        y = Y[sample[k],:]
        C = cost_fft2(u,y)
        # Mise à jour de la valeur de fft_u
        grad = - gamma/((k+1)**c) * W *  grad_heps_fft2(u, C, eps).to_array()
        fft_u = fft_u + pysh.shclasses.SHCoeffs.from_array(grad)  
        # Mise à zéro de son intégrale
        u = fft_u.expand()
        theta = np.pi/2 - u.lats()*np.pi/180
        u.data = u.data - np.mean(u.data * np.sin(theta).reshape(len(theta),1))
        fft_u = u.expand()
    return(u)


    

#############################################################################
# Functions for the entropic maps for F and Q 
#############################################################################

def ddprime(grid,y):
    '''
    - intermediate function for computing spherical barycentric projection
    '''
    theta = grid.lats()*np.pi/180 # in [-pi/2,pi/2]
    phi = grid.lons()*np.pi/180 # in [0,2pi]
    theta = np.pi/2 - theta  # in [0,pi]
    phi = phi - 2*np.pi*(1*(phi>np.pi)) # in [-pi,pi]
    LON,LAT = np.meshgrid(phi,theta)
    
    scal_prod_matrix = np.cos(LON)*np.sin(LAT)*y[0] + np.sin(LON)*np.sin(LAT)*y[1] + np.cos(LAT)*y[2]
    # https://fr.wikipedia.org/wiki/Coordonnées_sphériques.
    #scal_prod_matrix = np.minimum(1, scal_prod_matrix)
    #scal_prod_matrix = np.maximum(-1, scal_prod_matrix)
    d = np.arccos(scal_prod_matrix)
    d = - d / np.sqrt(1 - scal_prod_matrix**2)
    return(d)

def func_grid_3D(u):
    # Grid of points in 3D associated with u 
    tht = u.lats()*np.pi/180 #from degrees to radians
    phi = u.lons()*np.pi/180  
    phi,tht = np.meshgrid(phi,tht) 
    grid_3D = polar2cartesian(phi,tht)
    grid_3D = grid_3D.reshape((3,len(u.lons())*len(u.lats()))).T # uniform grid in cartesian coordinates
    return(grid_3D)
    



def Fentropic2(y,u,eps):
    '''
    Returns the distribution function F(y)
    - y is any point from the nu distribution
    - u is the first Kantorovich potential
    - eps is the regularization parameter
    '''
    grid_3D = func_grid_3D(u).T.reshape(3,len(u.lats()),len(u.lons()))
    
    # Cartesian gradient of the smooth c-transform of u(y)
    C = cost_fft2(u,y)
    to_exp = (u.to_array() - C )/eps
    MM = np.max(to_exp)
    Grad_u_ce = np.exp( to_exp - MM) #same size as C
    theta = u.lats()*np.pi/180
    theta = np.pi/2 - theta
    Grad_u_ce = Grad_u_ce * np.sin(theta).reshape(len(theta),1)
    argD = np.mean(Grad_u_ce)
    Grad_u_ce = grid_3D * ddprime(u,y) * Grad_u_ce / argD     
    Grad_u_ce = np.mean(Grad_u_ce.reshape(3,len(u.lats())*len(u.lons())),axis=1)
    # Riemannian gradient of the smooth c-transform of u(y)
    XXT = np.matmul( y.reshape(3,1), y.reshape(1,3))
    Riemangrad = (np.eye(3) - XXT) @ Grad_u_ce
    # Exponential map
    norm = np.linalg.norm(Riemangrad)
    Fy = np.cos(norm) * y - np.sin(norm)* Riemangrad/norm
    return(Fy)




# functions built from the spherical harmonic coefficients of the first Kantorovich potential, to retrieve u as a function or the quantile map Q.

def u_serie(points,u):
    ''' 
    Returns the spherical fourier serie of u for any points.
    - u shall be a SHGrid object giving u(x_i) for any x_i in a uniform grid on the 2-sphere.
    - points in 3D coordinates
    '''
    sh = sc.SphericalHarmonics(l_max=u.lmax, normalized=False)
    sh_values, sh_grads = sh.compute_with_gradients(points)
    fft_u = u.expand(normalization="ortho",csphase=1).to_array()
    
    res_u = []
    for i in range(sh_values.shape[0]):
        #if ((i % 1000)==0):
        #    print(i,np.round((time()-t0)/60,2),"minutes")
        Phi_i = sh_values[i] # take the spherical harmonics in R^3 for x_i
        uxi = 0
        for l in range(u.lmax+1):
            for m in range(u.lmax+1): # for all \m\>0,
                uxi = uxi + fft_u[0,l,m] * Phi_i[l*(l+1)+m] #m>0,
                uxi = uxi + fft_u[1,l,m] * Phi_i[l*(l+1)-m] #m<0,
        #uxi = uxi
        res_u.append(uxi)
    res_u = np.array(res_u)
    return(res_u)

def Qentropic(points,u):
    '''
    - points in 3D coordinates
    - u the Kantorovich potential of class SHgrid
    Returns the image of points from the entropic map, calculated by explicit gradients of spherical harmonics.
    '''
    sh = sc.SphericalHarmonics(l_max=u.lmax, normalized=False)
    sh_values, sh_grads = sh.compute_with_gradients(points)
    fft_u = u.expand(normalization="ortho",csphase=1).to_array()

    res = []
    for i in range(sh_grads.shape[0]):
        #if ((i % 1000)==0):
        #    print(i,np.round((time()-t0)/60,2),"minutes")
        DPhi_i = sh_grads[i] # gradients of spheric harmonics, shape (3,(lmax+1)**2)
        # Step 1) euclidean gradient of u in R^3
        grad = np.zeros(3) 
        for l in range(u.lmax+1):
            for m in range(u.lmax+1): # for all \m\>0,
                grad = grad + fft_u[0,l,m] * DPhi_i[:,l*(l+1)+m] #m>0,
                grad = grad + fft_u[1,l,m] * DPhi_i[:,l*(l+1)-m] #m<0

        # Step 2) Riemannian gradient of u through orthogonal projection onto the tangent space at xi. 
        xi = points[i]
        XXT = np.matmul( xi.reshape(3,1), xi.reshape(1,3))
        Riemangrad = (np.eye(3) - XXT) @ grad
        # Step 3) Exponential map, from the tangent space to the sphere 
        norm = np.linalg.norm(Riemangrad)
        Qxi = np.cos(norm) * xi - np.sin(norm)* Riemangrad/norm
        res.append(Qxi)

    res = np.array(res)
    return(res)

def QentropicBP_pts(points,u,data,eps):
    '''
    - points in 3D coordinates
    - u the first Kantorovich potential of class SHgrid
    - eps is the regularization parameter
    Returns the image of points from the entropic map, calculated by barycentric projection.
    '''
    # LogSumExp via the discretization of the sphere inherent to the spherical harmonics
    u_ce = []
    for z in data:
        C = cost_fft2(u,z)
        u_ce.append( c_transform_fft2(u, C, eps) )
    u_ce = np.array(u_ce)

    Q_points = []
    for x in points:
        Q_points.append(QentropicBP2(x,data,eps,u_ce))
    Q_points = np.array(Q_points)
    return(Q_points)

def QentropicBP2(x,data,eps,u_ce):
    '''
    Returns the quantile function Q(x) of a single x by barycentric projection
    - x is any point from the unit shere
    - u is the first Kantorovich potential
    - eps is the regularization parameter
    '''
    scal_prod_matrix = data[:,0] * x[0] + data[:,1] * x[1] + data[:,2] * x[2]
    cost_matrix = 0.5*np.arccos( scal_prod_matrix )**2
    dd_prime_x = - np.arccos(scal_prod_matrix) / np.sqrt( 1 - scal_prod_matrix**2 )

    #argD = np.mean( np.exp((u.to_array() - cost_fft2(u,x) )/eps) ) 
    arg = ( u_ce - cost_matrix )/eps 
    M = np.max(arg) #to avoid computational issues
    g_eps = np.exp( arg - M )/np.exp(-M)
    g_eps = g_eps / np.mean(g_eps)

    Grad_u = data.T * g_eps * dd_prime_x
    Grad_u = np.mean(Grad_u,axis=1)

    # Riemannian gradient of the smooth c-transform of u(y)
    XXT = np.matmul( x.reshape(3,1), x.reshape(1,3))
    Riemangrad = (np.eye(3) - XXT) @ Grad_u
    # Exponential map
    norm = np.linalg.norm(Riemangrad)
    Qx = np.cos(norm) * x - np.sin(norm)* Riemangrad/norm

    return(Qx)


#############################################################################
# Reference contours are contours of fixed latitude, oriented towards F_thetaM
#############################################################################
def rotateDataBack(rotateddata,k):
    Newdata = []
    for x in rotateddata:
        Newdata.append( RodriguesRot(x,-k,np.pi) )
    Newdata = np.array(Newdata)
    return(Newdata)

def contour_unif_rotated(MeanData,tau,size):
    k = MeanData+[0,0,1]
    k = k/np.linalg.norm(k)
    contourU = rotateDataBack(contour_unif(tau,size=size),k)
    return(contourU)

def signcurve_unif_rotated(MeanData,LON,size):
    k = MeanData+[0,0,1]
    k = k/np.linalg.norm(k)
    scU = rotateDataBack(signcurve_unif(LON,size=size),k)
    return(scU)


def scale_curve(u,rayons,F_thetaM,eps):
    ''' naive implementation of the volumes by way of average of samples '''
    nR = len(rayons)
    vol = -np.ones(nR)
    for c in range(nR):
        tau = rayons[c]

        nb = 2000
        grid_3D = np.random.normal(size=(nb,3))
        grid_3D = grid_3D / np.linalg.norm(grid_3D,axis=1).reshape(nb,1)

        indic = -np.ones(grid_3D.shape[0])
        for i,x in enumerate(grid_3D):
            indic[i] = 1*(Fentropic2(x,u,eps)@F_thetaM >= 1-2*tau)

        vol[c] = np.mean(indic)
    return(vol)




