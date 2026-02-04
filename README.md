# Spherical Quantiles

Codes used for the figures of the paper **Monge-Kantorovich quantiles for spherical data**, currently available on  

https://arxiv.org/pdf/2407.02085

## Codes available in python

Python 3.10.8 

Packages required: 
- pyshtools==4.10
- pot==0.9.0
- geomstats==2.7.0 
- numpy=1.26 

A full list is available in requirements.txt. 
You can also install dependencies with: 
`pip install -r requirements.txt `
(but this may include unnecessary packages).

## Notebooks for each experiment

The file functions.py contains the required functions.

Each notebook contains codes for some figures in the paper, as follows. 

Tuto: to compute regularized quantile function, hence contours and sign curves. Also introduces the implementation of our stochastic algorithm, with checks of convergence. 

RegVsUnreg: Comparison of regularized versus unregularized quantiles. 

VolumesRegion: contains codes for volumes (and scale curves) of regularized quantile regions. 

OtherQuantiles: Visual comparison of Mahalanobis/Spatial/MK quantiles. 

Classif: codes related to depth-based classification. 


