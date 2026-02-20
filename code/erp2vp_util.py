import random
from typing import Any
import cv2
import numpy as np
from PIL import Image


vp_positions = [
    (0, 90), 
    (0, 45), (60, 45), (120, 45), (180, 45), (240, 45), (300, 45), 
    (0, 0), (45, 0), (90, 0), (135, 0), (180, 0), (225, 0), (270, 0), (315, 0),
    (0, -45), (60, -45), (120, -45), (180, -45), (240, -45), (300, -45), 
    (0, -90)
]

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class VPExtractor:
    def __init__(self, FOV=45, lonlats=vp_positions, VPSize=(128, 128), img_shape=(512, 1024)):
        self.FOV = FOV
        self.lonlats = lonlats
        self.VPSize = VPSize
        self.img_shape = img_shape

        self.XY_list = []
        height, width = self.VPSize
        for lonlat in self.lonlats:
            THETA, PHI = lonlat
            f = 0.5 * width * 1 / np.tan(0.5 * self.FOV / 180.0 * np.pi)
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
            K_inv = np.linalg.inv(K)
            x = np.arange(width)
            y = np.arange(height)
            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)
            xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
            xyz = xyz @ K_inv.T
            
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            x_axis = np.array([1.0, 0.0, 0.0], np.float32)
            R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
            R = R2 @ R1
            xyz = xyz @ R.T
            lonlat = xyz2lonlat(xyz) 
            XY = lonlat2XY(lonlat, shape=self.img_shape).astype(np.float32)
            self.XY_list.append(XY)
    
    def __call__(self, erp, rotate=False):
        vp_list = []
        if rotate:
            self.gen_XY_list(45.*random.uniform(-1, 1))
        for XY in self.XY_list:
            vp = cv2.remap(erp, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            vp_list.append(vp)
        
        return vp_list
    
    def gen_XY_list(self, random_theta):
        self.XY_list = []
        height, width = self.VPSize
        for lonlat in self.lonlats:
            THETA, PHI = lonlat
            THETA += random_theta
            f = 0.5 * width * 1 / np.tan(0.5 * self.FOV / 180.0 * np.pi)
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
            K_inv = np.linalg.inv(K)
            x = np.arange(width)
            y = np.arange(height)
            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)
            xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
            xyz = xyz @ K_inv.T
            
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            x_axis = np.array([1.0, 0.0, 0.0], np.float32)
            R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
            R = R2 @ R1
            xyz = xyz @ R.T
            lonlat = xyz2lonlat(xyz) 
            XY = lonlat2XY(lonlat, shape=self.img_shape).astype(np.float32)
            self.XY_list.append(XY)

class Equirectangular:
    # The viewport extraction code is referenced from https://github.com/NitishMutha/equirectangular-toolbox
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp
    
