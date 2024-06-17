import cv2
import cv2.gapi
import h5py
import torch
from types import SimpleNamespace
import pnp
from pnp_algo import *
from typing import Dict

class Camera:
    def __init__(self, param):
        fx, fy, cx, cy = param
        self.c = param[2:4]
        self.f = param[0:2]

    def get_intrinsic_matrix(self):
        fx, fy = self.f
        cx, cy = self.c
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class H5py_file_reader:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')

    def get_group(self, name):
        group = self.f[name]
        dic = {}
        for k, v in group.items():
            data = np.array(v.__array__())
            dic[k] = torch.from_numpy(data)

        return dic 

def rbd(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }

class PnpSolver:
    def __init__(self, intrisics):
        self.camera = Camera(intrisics)

    ## qurey_features: output of feature extraction of query image, from extract_features(...)
    ## group: h5py group of ref image
    ## pred: output of feature matching, pred = matcher(inputs)
    ## ref_pose: pose of ref image from trajectory.txt
    def solve(self, qurey_features, group, pred):
        # qurey_features = rbd(qurey_features)
        pred = rbd(pred)
        ## matches.keys=dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])
        matches = SimpleNamespace(**pred)

        ## get 2d points from query image
        p2d = qurey_features['keypoints'][matches.matches[:, 0]]
        p2d = p2d / 0.4166666666666667

        #print("2D Keypoints")
        #print(p2d)

        ## get 3d points from ref image
        p3d = group['p3d'][matches.matches[:, 1]]

        assert len(p2d) == len(p3d)

        with open("p3d1.txt", "w") as f:
            for i in range(len(p3d)):
                f.write("i = " + str(i) + ",  " + str(p2d[i]) + ",  " + str(p3d[i]) + "\n")

        ## solve pnp
        #print("P3D shape - ", p3d.shape)
        #print("P2D shape - ", p2d.shape)

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(p3d.numpy(), p2d.numpy(), self.camera.get_intrinsic_matrix(), None)

        ## recover rotation matrix and translation.
        ## [P]c = rmat [P]w + tvec
        np_rodrigues = np.asarray(rvec[:,:],np.float64)

        C = torch.from_numpy(self.camera.get_intrinsic_matrix()).type(torch.float)
        R, t, num_inliers = pnp.ransac_pnp(C, p2d, p3d, k=6, max_it=500, inlier_err=30)

        rmat, _ = cv2.Rodrigues(np_rodrigues)
        print("rmat_shape = ", rmat.shape, ", tvec shape = ", tvec.shape)

        camera_position = -rmat.T @ tvec
        rotation_matrix = rmat.T

        t = -R.T @ t
        R = R.T

        print("camera shape", camera_position.shape)

        print(f"camera_position={camera_position} vs Lucas {t}")
        print(f"rotation_matrix={rotation_matrix} vs Lucas {R}")
        print(f"camera=", self.camera.get_intrinsic_matrix())

        ## calcualte score
        ratio = len(inliers) / len(p2d)

        return rotation_matrix, camera_position[:,0], ratio,

## use our pnp solver
def pnp_solve(intrinsic, qurey_features: Dict[str,torch.Tensor], group: Dict[str,torch.Tensor], pred: Dict[str,torch.Tensor]):
    c = intrinsic[2:4]
    f = intrinsic[0:2]

    # qurey_features = rbd(qurey_features)
    #pred = rbd(pred)
    ## matches.keys=dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])


    ## get 2d points from query image
    p2d = qurey_features['keypoints'][pred["matches"][:, 0]]

    ## get 3d points from ref image
    p3d = group['p3d'][pred["matches"][:, 1]]
    n = len(p2d)

    if n < 8:
        print("Not enough matches to compute matrix")
        return torch.zeros((3,3)), torch.zeros(3), 0.0

    assert len(p2d) == len(p3d)
    ## solve pnp

    rmat, tvec, inlier_cnt = pnp_ransac(p3d, p2d, c, f[0])

    print(f"rvec={rmat}")
    print(f"tvec={tvec}")
    print("Inliers = ", inlier_cnt, "/", len(p2d))
    print("Max kp", p2d.max())
    print(tvec)

    ## calcualte score
    ratio = inlier_cnt / len(p2d)

    return rmat, tvec, ratio






