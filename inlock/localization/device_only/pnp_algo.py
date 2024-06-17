import torch
import numpy as np
import cmath
from typing import Tuple
import svd

## takes in 6 points and solve for least square problem.
## Pmat transform homogeneous 3d points to 2d points.
## p2d are at z = 1, camera is at origin facing +z direction.
def pnp(p2d, p3d):
    assert p2d.shape[0] == 6
    assert p3d.shape[0] == 6
    A = torch.zeros(18, 12)
    b = torch.zeros(18, 1)
    for i in range(6):
        x, y, z = p3d[i]
        u, v = p2d[i]
        xx = torch.tensor([x, y, z, 1])
        A[i * 3, 4:8] = -xx
        A[i * 3, 8:12] = v * xx
        A[i * 3 + 1, 0:4] = xx
        A[i * 3 + 1, 8:12] = -u * xx
        A[i * 3 + 2, 0:4] = -v * xx
        A[i * 3 + 2, 4:8] = u * xx

    res = torch.linalg.lstsq(A, b)
    P = res.solution
    Pmat = P.reshape(3, 4)  ## row major so correct
    return Pmat


def pnp2(p2d, p3d):
    n = p2d.shape[0]
    A = torch.zeros(3 * n, 12)
    # b = torch.zeros(18, 1)
    for i in range(n):
        x, y, z = p3d[i]
        u, v = p2d[i]
        xx = torch.tensor([x, y, z, 1])
        A[i * 3, 4:8] = -xx
        A[i * 3, 8:12] = v * xx
        A[i * 3 + 1, 0:4] = xx
        A[i * 3 + 1, 8:12] = -u * xx
        A[i * 3 + 2, 0:4] = -v * xx
        A[i * 3 + 2, 4:8] = u * xx

    ## subject to constraint P.norm() = 1 since only up to scale
    ## eigen vector with least eigen value gives the least square
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1]  ## = last col of V
    Pmat = P.reshape(3, 4)  ## row major so correct
    return Pmat


def q2rmat(q):
    w, x, y, z = q
    return torch.tensor([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


def rand_rmat():
    # Random quaternion
    q = torch.randn(4)
    q = q / torch.norm(q)  # Normalize the quaternion

    # Quaternion to rotation matrix
    w, x, y, z = q
    rmat = q2rmat(q)
    return rmat


def rand(a, b, size):
    return torch.rand(size) * (b - a) + a


def sq(a):
    return torch.square(a)

def roots2(a:complex,b:complex,c:complex) -> Tuple[complex, complex]:
    bp:complex=complex(b/2)
    delta:complex=complex(bp*bp-a*c)
    u1:complex=(-bp-cmath.sqrt(delta))/complex(a)
    u2:complex=-u1-b/a
    return u1,u2

"""
Direct adapation of https://math.stackexchange.com/questions/785/is-there-a-general-formula-for-solving-quartic-degree-4-equations
"""
@torch.jit.script
def Cardano(a,b,c,d) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    J: complex = -0.5 + 0.8660254037844388j  # torch.exp(2j * torch.pi / 3)
    Jc: complex = -0.5 - 0.8660254037844388j

    z0=b/3/a
    a2,b2 = a*a,b*b
    p=-b2/3/a2 +c/a
    q=(b/27*(2*b2/a2-9*c/a)+d)/a
    D=-4*p*p*p-27*q*q
    r=cmath.sqrt(-D/27+0j)
    u=((-q-r)/2)**0.33333333333333333333333
    v=((-q+r)/2)**0.33333333333333333333333
    w=u*v
    w0=abs(w+p/3)
    w1=abs(w*J+p/3)
    w2=abs(w*Jc+p/3)
    if w0<w1:
      if w2<w0 : v*=Jc
    elif w2<w1 : v*=Jc
    else: v*=J
    return u+v-z0, u*J+v*Jc-z0, u*Jc+v*J-z0

@torch.jit.script
def Roots_2(a,b,c) -> Tuple[torch.Tensor,torch.Tensor]:
    bp=b/2
    delta=bp*bp-a*c
    r1=(-bp-delta**.5)/a
    r2=-r1-b/a
    return r1,r2

@torch.jit.script
def ferrari(a,b,c,d,e):
    "Ferrarai's Method"
    "resolution of P=ax^4+bx^3+cx^2+dx+e=0, coeffs reals"
    "First shift : x= z-b/4/a  =>  P=z^4+pz^2+qz+r"
    z0=b/4/a
    a2,b2,c2,d2 = a*a,b*b,c*c,d*d
    p = -3*b2/(8*a2)+c/a
    q = b*b2/8/a/a2 - 1/2*b*c/a2 + d/a
    r = -3/256*b2*b2/a2/a2 +c*b2/a2/a/16-b*d/a2/4+e/a
    "Second find y so P2=Ay^3+By^2+Cy+D=0"
    A=torch.tensor(8, dtype=torch.complex64)
    B=-4*p
    C=-8*r
    D=4*r*p-q*q
    y0,y1,y2=Cardano(A,B,C,D)
    if abs(y1.imag)<abs(y0.imag): y0=y1
    if abs(y2.imag)<abs(y0.imag): y0=y2
    a0=(-p+2*y0)**.5
    if a0==0 : b0=y0**2-r
    else : b0=-q/2/a0
    r0,r1=Roots_2(torch.tensor(1,dtype=torch.complex64),a0,y0+b0)
    r2,r3=Roots_2(torch.tensor(1,dtype=torch.complex64),-a0,y0-b0)
    return (r0-z0,r1-z0,r2-z0,r3-z0)

one = torch.tensor(1., dtype=torch.complex64)
print(Cardano(one,one,one,one))
print(ferrari(one,one,one,one,one))

def f(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.complex64)







@torch.jit.script
def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = torch.reshape(1 / N * (torch.sum(A, dim=1)), (3, 1))
    B_centroid = torch.reshape(1 / N * (torch.sum(B, dim=1)), (3, 1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = torch.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + torch.outer(ai, bi)

    #U, S, V_transpose = torch.linalg.svd(H)

    U, S, V_transpose = svd.svd(H)
    V_transpose = V_transpose.T

    V = V_transpose.T
    U_transpose = U.T

    x = torch.ones(3)

    x[2] = svd.det3x3(V) * svd.det3x3(U_transpose)

    R = V @ torch.diag(x) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t

## p2d are normalised z = 1
## outputs R and t such that Pw = R * Pc + t
@torch.jit.script
def pnp_grunert(p3d, p2d):
    p1 = p3d[0]
    p2 = p3d[1]
    p3 = p3d[2]
    p4 = p3d[3]

    # print("p1:\n", p1)
    # print("p2:\n", p2)
    # print("p3:\n", p3)
    # print("p4:\n", p4)

    a = (p2 - p3).norm()
    b = (p3 - p1).norm()
    c = (p1 - p2).norm()

    a2 = sq(a)
    b2 = sq(b)
    c2 = sq(c)

    u1 = p2d[0][0].item(); v1 = p2d[0][1].item()
    u2 = p2d[1][0].item(); v2 = p2d[1][1].item()
    u3 = p2d[2][0].item(); v3 = p2d[2][1].item()
    u4 = p2d[3][0].item(); v4 = p2d[3][1].item()

    q1 = torch.ones(3)
    q1[0] = u1
    q1[1] = v1

    q2 = torch.ones(3)
    q2[0] = u2
    q2[1] = v2

    q3 = torch.ones(3)
    q3[0] = u3
    q3[1] = v3

    #q4 = torch.tensor([u4, v4, 1.])

    # print("q1:\n", q1)
    # print("q2:\n", q2)
    # print("q3:\n", q3)

    j1 = q1 / q1.norm()
    j2 = q2 / q2.norm()
    j3 = q3 / q3.norm()

    cos_a = j2.dot(j3)
    cos_b = j3.dot(j1)
    cos_c = j1.dot(j2)

    frac_1 = (a2 - c2) / b2
    frac_2 = (a2 + c2) / b2
    frac_3 = (b2 - c2) / b2
    frac_4 = (b2 - a2) / b2

    A4 = sq(frac_1 - 1.0) - 4 * c2 / b2 * sq(cos_a)

    A3 = 4 * (frac_1 * (1 - frac_1) * cos_b - (1 - frac_2) * cos_a * cos_c + 2 * c2 / b2 * cos_a * cos_a * cos_b)

    A2 = 2 * (sq(frac_1) - 1 + 2 * sq(frac_1) * sq(cos_b) + 2 * frac_3 * sq(
        cos_a) - 4 * frac_2 * cos_a * cos_b * cos_c + 2 * frac_4 * sq(cos_c))

    A1 = 4 * (- frac_1 * (1 + frac_1) * cos_b + 2 * a2 / b2 * sq(cos_c) * cos_b - (1 - frac_2) * cos_a * cos_c)

    A0 = sq(1 + frac_1) - 4 * a2 / b2 * sq(cos_c)

    A = torch.tensor([A4, A3, A2, A1, A0])

    roots = ferrari(f(A4),f(A3),f(A2),f(A1),f(A0))
    #roots(A.numpy())

    best_error = 1e9
    best_R = torch.zeros((3,3))
    best_t = torch.zeros((1,3))

    for v in roots:
        #v = torch.tensor(v)
        if v.is_complex():
            v = v.abs()

        up = (-1 + frac_1) * sq(v) - 2 * frac_1 * cos_b * v + 1 + frac_1
        down = 2 * (cos_c - v * cos_a)
        u = up / down

        # s1 = c2 / (1 + sq(u) - 2 * u * cos_c)

        s12 = b2 / (1 + sq(v) - 2 * v * cos_b)
        s1 = torch.sqrt(s12)

        s2 = u * s1
        s3 = v * s1

        P1 = s1 * j1
        P2 = s2 * j2
        P3 = s3 * j3

        A = torch.stack([P1, P2, P3], dim=0).T  ## image space
        B = torch.stack([p1, p2, p3], dim=0).T  ## world space

        R, t = arun(A, B)  ## R, t is pose of camera

        # print("R:\n", R)
        # print("t:\n", t)
        q4 = torch.ones(3)
        q4[0] = u4
        q4[1] = v4

        t = t.reshape(3)
        camera_space_guess = (p4 - t) @ R  ## guess where p4 should be in image space
        img_plane_coord_guess = camera_space_guess / camera_space_guess[2]  ## normalise to z = 1

        # print("img_plane_coord_guess:\n", img_plane_coord_guess)
        # print("q4:\n", q4)

        error = torch.norm(img_plane_coord_guess - q4)

        if best_error > error:
            best_error = error
            best_R = R
            best_t = t
    return best_R, best_t

#pnp_grunert(torch.zeros(4,3), torch.zeros(4,2))

## p3d      N x 3, world coordinates with respect to navvis
## p2d_i    N x 2, pixel coordinates
## c        1 x 2,
## f        1 x 1,
## outputs R and t such that Pw = R * Pc + t
def pnp_ransac(p3d, p2d_i, c, f) -> Tuple[torch.Tensor, torch.Tensor, int]:
    ## constants to set
    n_itr = 1000
    n_sample = 4
    pixel_threshold = 2.0  ## this is the pixel difference on image.
    threshold = torch.tensor(pixel_threshold / f)

    ## normalise all 2d points to z = 1
    p2d = (p2d_i - c) / f

    ## ransac loop
    most_inliers = 0
    best_R = torch.zeros((3,3))
    best_t = torch.zeros((1,3))

    for _ in range(n_itr):
        ## randomly sample n_sample points
        idx = torch.randperm(p3d.shape[0])[:n_sample]
        p3d_sample = p3d[idx]
        p2d_sample = p2d[idx]

        R, t = pnp_grunert(p3d_sample, p2d_sample)

        ## calculate inliers
        ## back project 3d points to 2d points and compare with p2d
        p2d_guess = (p3d - t) @ R
        p2d_nomalised = p2d_guess / p2d_guess[:, 2][..., None]

        diff = torch.norm(p2d_nomalised[:, :2] - p2d, p=2, dim=1)

        inliers = (diff < threshold).sum()

        if inliers > most_inliers:
            most_inliers = inliers
            best_R = R
            best_t = t

    inlier_cnt = most_inliers
    return best_R, best_t, inlier_cnt


def test_2():
    n = 4
    p2d = rand(-10, 10, (n, 2))
    z = rand(2, 10, (n, 1))
    p3d_c = torch.cat([p2d * z, z], dim=1)

    # print("p3d_c:\n", p3d_c)

    R = rand_rmat()
    t = rand(-10, 10, (1, 3))
    p3d = p3d_c @ R.T + t

    # print("R:\n", R)
    # print("t:\n", t)

    sR, st = pnp_grunert(p3d, p2d)

    # print("sR:\n", sR)
    # print("st:\n", st)


def test_1():
    ## travial 6 points to test if pose is correct
    n = 8
    p2d = rand(-10, 10, (n, 2))
    z = rand(2, 10, (n, 1))
    p3d_c = torch.cat([p2d * z, z], dim=1)

    R = rand_rmat()
    t = rand(-10, 10, (1, 3))
    p3d = p3d_c @ R.T + t

    Pmat = pnp2(p2d, p3d)

    print("p2d:\n", p2d)
    print("p3d:\n", p3d)

    print("R:\n", R)
    print("t:\n", t)
    print("Pmat:\n", Pmat)

    hp3d = torch.cat([p3d, torch.ones(n, 1)], dim=1)
    hp2d = hp3d @ Pmat.T

    print("hp2d:\n", hp2d)
    hp2d = hp2d / hp2d[:, 2].unsqueeze(1)

    print("after normalisation, \n hp2d:\n", hp2d)

    pR = Pmat[:, 0:3]
    pt = Pmat[:, 3]

    print("R.T@R:\n", pR.transpose(0, 1) @ pR)

    ## additional test
    print("additional test")
    n = 8
    p2d = rand(-100, 100, (n, 2))
    z = rand(10, 30, (n, 1))
    p3d_c = torch.cat([p2d * z, z], dim=1)
    p3d = p3d_c @ R.T + t
    hp3d = torch.cat([p3d, torch.ones(n, 1)], dim=1)
    hp2d = hp3d @ Pmat.T
    hp2d = hp2d / hp2d[:, 2].unsqueeze(1)
    print("p2d:\n", p2d)
    print("hp2d:\n", hp2d)


def test_3():
    n = 300
    p2d = rand(-1, 1, (n, 2))
    z = rand(40, 960, (n, 1))
    p3d_c = torch.cat([p2d * z, z], dim=1)

    R = rand_rmat()
    t = rand(-10, 10, (1, 3))
    p3d = p3d_c @ R.T + t

    print("R:\n", R)
    print("t:\n", t)

    c = torch.tensor([0, 0])
    f = 1

    sR, st, inliers = pnp_ransac(p3d, p2d, c, f)

    print("sR:\n", sR)
    print("st:\n", st)
    print(inliers)


def test_4():
    n = 300
    p2d = rand(-1, 1, (n, 2))
    z = rand(40, 960, (n, 1))
    p3d_c = torch.cat([p2d * z, z], dim=1)

    R = rand_rmat()
    t = rand(-10, 10, (1, 3))
    p3d = p3d_c @ R.T + t

    print("R:\n", R)
    print("t:\n", t)

    c = torch.tensor([0, 0])
    f = 1

    ## add noise to p3d
    p3d = p3d + rand(-5, 5, (n, 3))

    sR, st, inliners = pnp_ransac(p3d, p2d, c, f)

    print("sR:\n", sR)
    print("st:\n", st)
    print("inliners:\n", inliners)


if __name__ == "__main__":
    test_4()