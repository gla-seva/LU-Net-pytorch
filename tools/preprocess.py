import os
import numpy as np
import open3d as o3d
import pandas as pd

os.chdir("/home/vglasov/Reseach/LU-Net-pytorch/")
import config

class Pcd2ImageTransform:
    def __init__(self, shape=config.shape, angular=config.angular, rand_angle=False):
        self.height, self.width = shape
        self.rand_angle = rand_angle
        y_angular, x_angular = angular
        self.x_delta, self.y_delta = x_angular / self.width, y_angular / self.height
    
    def fit(self, pcd, label_list):
        """
        Input: 
            pcd = PointCloud file, 
            label_list = list of each box property
        Output:
            self
        """
        
        if pcd.is_empty():
            raise RuntimeError("Point Cloud must be non-empty")
            
        self.pcd = pcd
        x, y, z = np.array(self.pcd.points).T
        r = np.sqrt(x**2 + y**2 + z**2)
        
        random_azimuth_shift = np.random.uniform(-np.pi, np.pi) if self.rand_angle else 0.
        
        azimuth_angle = (np.arctan2(x, y) + random_azimuth_shift) % (2 * np.pi)
        elevation_angle = np.arcsin(z / r)
        
        x_img = np.floor(azimuth_angle / self.x_delta).astype(int)
        y_img = np.floor(elevation_angle / self.y_delta).astype(int)
        
#         x_img -= x_img.min()
        y_img -= y_img.min()
        
        self.pcd_labels = np.zeros(len(self.pcd.points), dtype=int)
        for x_min, y_min, z_min, x_max, y_max, z_max in label_list:            
            self.pcd_labels[np.all([x_min <= x,
                                    y_min <= y, 
                                    z_min <= z,
                                    x <= x_max,
                                    y <= y_max,
                                    z <= z_max], axis=0)] = 1
        
        self.transformation = pd.DataFrame({'x': x, 
                                            'y': y,
                                            'z': z,
                                            'x_img': x_img,
                                            'y_img': y_img,
                                            'r': r,
                                            'label': self.pcd_labels})        
        return self
    
    def transform(self):
        """
        Input: 

        Output:
            range_image - rangle image with sizes Height * Width * 2 (range, elevation)
            mask - mask with existing pixels
            labels - mask with pixels to predict
        """
        
        if not hasattr(self, 'transformation'):
            raise RuntimeError("Call fit() method first.")
            
        a, b = np.meshgrid(range(0, config.shape[0]), range(0, config.shape[1]))
        template = pd.DataFrame({'y_img': a.reshape(-1), 'x_img': b.reshape(-1)}).set_index(['y_img', 'x_img'])
        
        xyz = self.transformation.groupby(['y_img', 'x_img'])['x', 'y', 'z'].mean()
        r   = self.transformation.groupby(['y_img', 'x_img'])['r'].min()
        lb  = self.transformation.groupby(['y_img', 'x_img'])['label'].mean()
        
        X = template.join(xyz['x']).unstack(fill_value=0.).values[::-1, :]
        Y = template.join(xyz['y']).unstack(fill_value=0.).values[::-1, :]
        Z = template.join(xyz['z']).unstack(fill_value=0.).values[::-1, :]
        range_image = template.join(r).unstack(fill_value=0.).values[::-1, :]
#         mask = np.isfinite(template.join(X).unstack(fill_value=0.).values[::-1, :])
        labels = template.join(lb).unstack(fill_value=0.).values[::-1, :]
        
#         return range_image, mask, labels
        return np.nan_to_num(np.stack((X, Y, Z, range_image, labels), axis=-1))
    
    def fit_transform(self, pcd, label_list):
        self.fit(pcd, label_list)
        return self.transform()
    
    def inverse_transform(self, labels):
        """
        Input: 
            labels - predicted mask
        Output:
            PointCloud - points projected back with colored prediction
        """
        
        labels = labels[::-1, :]
        x_idx_pred, y_idx_pred = np.where(labels)
        prediction = np.zeros(len(self.transformation))
        for y, x in zip(*np.where(labels)):
            prediction[(self.transformation['x_img'] == x) & (self.transformation['y_img'] == y)] = 1
        pcd = o3d.PointCloud(self.pcd)
        pcd.colors = o3d.Vector3dVector(np.vstack((prediction,
                                                   np.zeros_like(prediction),
                                                   prediction)).T)
        return pcd
    
# if __name__ == "__main__":
#     pts_path = "data/example.pcd"
#     labels_path = "data/example.txt"

#     fragment = o3d.read_point_cloud(pts_path)
#     labels_list = [list(map(float, f.split()[4:10])) for f in open(labels_path, "r").readlines()]


#     pcd2img = Pcd2ImageTransform().fit(fragment, labels_list)
#     X, mask, labels = pcd2img.transform()

#     inversed_pcd = pcd2img.inverse_transform(labels)

