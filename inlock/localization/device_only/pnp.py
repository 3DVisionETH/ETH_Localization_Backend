import torch

def pnp(C, p2d, p3d):
    # Normalize kps
    p3d = torch.cat((p3d[:,:3], torch.ones(p3d.shape[0], 1)), dim=1)
    p2d = torch.cat((p2d[:,:2], torch.ones(p2d.shape[0], 1)), dim=1)

    C_inv = torch.linalg.inv(C)
    p2d = (C_inv @ p2d.T).T
    p2d = p2d / p2d[:,2:]

    p2d_avg = torch.mean(p2d, dim=0)
    p2d_scale = torch.mean(torch.abs(p2d - p2d_avg.unsqueeze(0)), dim=0)
    N2d = torch.tensor([
        [1.0/p2d_scale[0], 0.0, -p2d_avg[0]],
        [0.0, 1.0/p2d_scale[1], -p2d_avg[1]],
        [0.0, 0.0, 1.0]
    ])
    #N2d = torch.diag(torch.ones(3))
    p2d = (N2d @ p2d.T).T

    # Normalize position
    p3d_avg = torch.mean(p3d, dim=0)
    p3d_scale = torch.mean(torch.abs(p3d - p3d_avg.unsqueeze(0)), dim=0)
    N3d = torch.tensor([
        [1.0 / p3d_scale[0], 0.0, 0.0, -p3d_avg[0]],
        [0.0, 1.0 / p3d_scale[1], 0.0, -p3d_avg[1]],
        [0.0, 0.0, 1.0 / p3d_scale[2], -p3d_avg[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    p3d = (N3d @ p3d.T).T

    u = p2d[:,0].unsqueeze(1)
    v = p2d[:,1].unsqueeze(1)

    n = p2d.shape[0]

    zero = torch.zeros((n,4))

    mat = torch.zeros((2 * n, 12))
    mat[0::2] = torch.cat([p3d,zero,-u*p3d], dim=1)
    mat[1::2] = torch.cat([zero,p3d,-v*p3d], dim=1)

    U,S,Vt = torch.linalg.svd(mat, full_matrices=False)

    sol = torch.zeros((3,4))
    sol[:,:3] = torch.diag(torch.tensor([1.,1.,1.]))

    sol = Vt[11,:]
    sol = sol.reshape((3,4))
    sol = torch.linalg.inv(N2d) @ sol @ N3d

    R = sol[:,:3]
    sol = sol / torch.mean(torch.linalg.norm(R,dim=0))

    R = sol[:,:3]
    t = sol[:,3]

    U,S,Vt = torch.linalg.svd(R)

    if torch.det(R) < 0:
        R = U @ torch.diag(torch.tensor([1.,1.,-1.])) @ Vt
    else:
        R = U @ Vt

    return R, t

def inverse_pose(R, t):
    return R.T, R.T @ t

def ransac_pnp(C, p2d, p3d, k, inlier_err, max_it) -> (torch.Tensor, torch.Tensor, int):
    def reproj_error(R, t):
        p3d_rot = (R @ p3d.T)
        p = (C @ (p3d_rot + t.unsqueeze(1))).T
        p = p[:,:2] / p[:,2:3]
        return torch.linalg.norm(p - p2d, dim=1)

    best_R,best_t = torch.zeros((3,3)), torch.zeros(3)
    best_inliers = 0
    n = p2d.shape[0]

    for i in range(max_it):
        perm = torch.randperm(n)
        indices = perm[:k]

        R,t = pnp(C, p2d[indices], p3d[indices])

        err = reproj_error(R, t)
        inliers = (err < inlier_err).sum()
        #print("mean reproj ", err.mean(), inliers.item(), "/", n)

        if inliers > best_inliers:
            print("Num inliers = ", inliers, n)
            best_inliers = inliers
            best_R = R
            best_t = t

    return best_R, best_t, best_inliers


def basic_test():
    C = torch.tensor([
        [2.,0.,0.],
        [0.,2.,0.],
        [0.,0.,1.]
    ])

    points3d = torch.tensor([
        [-1,-1,1],
        [1,-1,1],
        [1,1,1],
        [-1,1,1],
        [-1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2],
        [0, 0, 1],
    ], dtype=torch.float)
    points2d = (C @ (points3d + torch.tensor([0.1,0,0.3]).unsqueeze(0)).T).T

    points2d = points2d[:,:2] / points2d[:,2:3]
    #print(points2d == points3d)

    R,t,inliers = ransac_pnp(C, points2d, points3d, k=len(points3d), inlier_err=0.1, max_it=10)
    R,t = inverse_pose(R, t)
    print("R=",torch.round(R, decimals=2))
    print("t=",torch.round(t, decimals=2))

if __name__ == "__main__":
    basic_test()