#!/usr/bin/env python

import numpy as np

#paths
dataset = "/home/vglasov/datasets/L_CAS_3D_Point_Cloud_People_Dataset/3D Point Cloud People Dataset/"
synthetic_dataset = "/home/vglasov/datasets/L_CAS_3D_Point_Cloud_People_Dataset/Synthetic 3D Point Cloud People Dataset/"

#Lidar image spec
eps = 1e-6
shape = (16, 384) #image shape
angular = (0.5236, 2 * np.pi) # approximation of 30º and 360º


