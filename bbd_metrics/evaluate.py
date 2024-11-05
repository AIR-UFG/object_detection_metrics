import json
from src.standalone  import *
import numpy as np
from scipy.spatial.transform import Rotation as R

def evaluate(gt_instance, prediction_instance):
    # formatting the ground truth
    center_gt = gt_instance["contour"]["center3D"]
    center = np.array([center_gt["x"], center_gt["y"], center_gt["z"]])
    
    rotation_gt = gt_instance["contour"]["rotation3D"]
    rotation = [rotation_gt["x"], rotation_gt["y"], rotation_gt["z"]]
    
    size_gt = gt_instance["contour"]["size3D"]
    size = [size_gt["x"], size_gt["y"], size_gt["z"]]
    
    # Bounding Box Ground Truth
    p = center
    r = R.from_euler('xyz', rotation, degrees=True).as_matrix() # euler angles
    d = np.array(size) # size
    T = np.vstack([np.column_stack([d*r,p]),[0,0,0,1]])
    bb = OBB(T)

    # formatting the prediction
    center_prediction = prediction_instance["center3D"]
    center = np.array([center_prediction["x"], center_prediction["y"], center_prediction["z"]])
    
    rotation_prediction = prediction_instance["rotation3D"]
    rotation = [rotation_prediction["x"], rotation_prediction["y"], rotation_prediction["z"]]
    
    size_prediction = prediction_instance["size3D"]
    size = [size_prediction["x"], size_prediction["y"], size_prediction["z"]]
    
    # Bounding Box Prediction
    p = center
    r = R.from_euler('xyz', rotation, degrees=True).as_matrix() # euler angles
    d = np.array(size) # size
    T = np.vstack([np.column_stack([d*r,p]),[0,0,0,1]])
    bb2 = OBB(T)

    # IoU
    iou_res = bb.IoU_v(bb2,1e-8)

    # V2V distance
    v2v_res = bb.v2v(bb2)

    # bbd
    bdb_res = bb.bbd(bb2)

    return iou_res, v2v_res, bdb_res






