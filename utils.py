import numpy as np
import matplotlib.pyplot as plt

def convolve_preserve(g, h, method='fill', value=0):
    ofs = h.shape[0] // 2
    tmp = np.ones((g.shape[0]+2*ofs, g.shape[1]+2*ofs))
    if method == 'fill':
        tmp *= value
        tmp[ofs:-ofs, ofs:-ofs] = g
        out = convolve(tmp, h)
        return out[ofs:-ofs, ofs:-ofs]
    elif method == 'mirror':
        tmp[ofs:-ofs, ofs:-ofs] = g
        tmp[:, 0:ofs] = tmp[:, ofs:2*ofs] # left
        tmp[:, -ofs:] = tmp[:, -2*ofs:-ofs] # right
        tmp[0:ofs:, ] = tmp[ofs:2*ofs:, ] # top
        tmp[-ofs:, ] = tmp[-2*ofs:-ofs, ] # bottom
        out = convolve(tmp, h)
        return out[ofs:-ofs, ofs:-ofs]
    else:
        # Nearest Neighbours
        tmp[:, 0:ofs] = array([tmp[:,ofs],]*ofs).transpose() # left
        tmp[:, -ofs:] = array([tmp[:,-ofs],]*ofs).transpose()  # right
        tmp[0:ofs:, ] = array([tmp[ofs,:],]*ofs) # top
        tmp[-ofs:, ] = array([tmp[-ofs,:],]*ofs) # bottom
        
        tmp[0,0] = tmp[ofs, ofs]
        tmp[0,-1] = tmp[ofs,-ofs]
        tmp[-1, 0] = tmp[-ofs, ofs]
        tmp[-1, -1] = tmp[-ofs, -ofs]
        
        out = convolve(tmp, h)
        return out[ofs:-ofs, ofs:-ofs]

def gaussian(sz, sigma):
    out = np.zeros((sz, sz))
    ofs = sz // 2
    for j in range(sz):
        for k in range(sz):
            m, n = j - ofs, k - ofs
            out[j, k] = np.exp(-(m*m + n*n)/(2*sigma*sigma))
    return out/out.sum()

def convolve(g, h, step=1):
    '''Note to self: Don't use np.mat with
       regular ndarrays. It results in
       bizarre performance. '''
    out = np.zeros((g.shape))
    k = h.shape[0]
    ofs = k // 2 
    for i in range(k, g.shape[0], step):
        for j in range(k, g.shape[1], step):
            sub_img = g[i-k:i, j-k:j]
            t = sub_img * h
            out[i-ofs, j-ofs] = t.sum()
    return out

def harris_response(I, sz=9, sigma=2):
    gauss = gaussian(sz, sigma)
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Iu = convolve(I, sobel)
    Iv = convolve(I, sobel.T)
    Iuu = convolve(Iu*Iu, gauss)
    Ivv = convolve(Iv*Iv, gauss)
    Iuv = convolve(Iu*Iv, gauss)
    denom = Iuu + Ivv + 1e-10
    num = Iuu*Ivv - Iuv*Iuv
    out = num / denom
    out[out == np.nan] = 0
    return out

def is_max(H, i, j):
    center_val = H[i, j]
    all_i = [i-1, i, i+1, i-1, i+1, i-1, i, i+1]
    all_j = [j-1, j-1, j-1, j, j, j+1, j+1, j+1]
    if np.all(H[all_i, all_j] < center_val):
        return True
    return False

def local_max(H):
    x = []
    y = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            try:
                if is_max(H, i, j):
                    x.append(i)
                    y.append(j)
            except IndexError as e:
                continue
    return x, y

def filter_local_max(H, x, y, n=100):
    '''x, y: lists/arrays of the local maxima's points'''
    y_sort = [Y for (Y, X) in sorted(zip(y, x),  key=lambda k:H[k[0], k[1]])]
    x_sort = [X for (Y, X) in sorted(zip(y, x),  key=lambda k:H[k[0], k[1]])]
    return y_sort[-n:], x_sort[-n:]

def adaptive_suppression(H, x, y, n=100, c=0.9):
    '''  Pretty garbage algorithm. Some
    sort of nearest-neighbor algorithm could be much
    more efficient. TODO: Add kd tree.'''
    dist = []
    for x0, y0 in zip(x, y):
        v = H[y0, x0]
        d = np.inf 
        for x1, y1 in zip(x, y):
            if c*H[y1, x1] > v:
                dd = abs(x1 - x0) + abs(y1 - y0) 
                if dd < d:
                    d = dd
        dist.append(d)
    ind = np.argsort(dist) 
    ind = ind[-n:]
    x = np.asarray(x)
    y = np.asarray(y)
    return y[ind], x[ind]

def unpack(ls):
    x = []; y = []
    for e in ls:
        x.append(e[0])
        y.append(e[1])
    return np.asarray(x), np.asarray(y)

def extract_descriptors(I, x, y, ks=21):
    '''x, y are lists of keypoints.'''
    assert len(x) == len(y)
    ofs = ks // 2
    out = []
    x_out = []
    y_out = []
    for xx, yy in zip(x, y):
        sub = I[xx-ofs:xx+ofs+1, yy-ofs:yy+ofs+1] 
        if sub.shape[0] == ks and sub.shape[1] == ks: 
            x_out.append(xx)
            y_out.append(yy)
            out.append(sub)
    return np.stack(out), np.asarray(y_out), np.asarray(x_out)

def d_hat(d):
    return (d - d.mean()) / np.std(d, ddof=1)

def err(d0, d1):
    out = (d1 - d0)**2     
    return out.sum()
        
def match_descriptors(d1, d2, x1, y1, x2, y2):
    out_1 = []
    out_2 = []
    for d, xx1, yy1 in zip(d1, x1, y1):
        d_h = d_hat(d)
        mn = np.inf
        jj = 0
        for dd, xx2, yy2 in zip(d2, x2, y2):
            dd_h = d_hat(dd)
            e = err(d_h, dd_h)
            if e < mn:
                mn = e
                jj = (xx2, yy2)
        out_1.append((xx1, yy1))
        out_2.append(jj)
    return np.asarray(out_1), np.asarray(out_2)


def match_descriptors_threshold(d1, d2, x1, y1, x2, y2, r=0.7):
    out_1 = []
    out_2 = []
    for d, xx1, yy1 in zip(d1, x1, y1):
        d_h = d_hat(d)
        mn = np.inf
        mn2 = np.inf
        jj = 0
        for dd, xx2, yy2 in zip(d2, x2, y2):
            dd_h = d_hat(dd)
            e = err(d_h, dd_h)
            if e < mn2:
                mn2 = e
                if e < mn:
                    mn2 = mn
                    mn = e
                    jj = (xx2, yy2)
        if mn < r*mn2:
            out_1.append((xx1, yy1))
            out_2.append(jj)
    return np.asarray(out_1), np.asarray(out_2)


def homography(pts1, pts2):
    # split coordinates into lists
    u1 = pts1[:,0]
    v1 = pts1[:,1]
    u2 = pts2[:,0]
    v2 = pts2[:,1]
    
    # initialize A matrix
    n = len(pts1)
    A = np.zeros(shape=(2*n,9))
    i = 0
    
    # add rows for each point match
    for j in range(0, 2*n, 2):
        A[j] = [0, 0, 0, -u1[i], -v1[i], -1, v2[i]*u1[i], v2[i]*v1[i], v2[i]]
        A[j+1] = [u1[i], v1[i], 1, 0, 0, 0, -u2[i]*u1[i], -u2[i]*v1[i], -u2[i]]
        i += 1
    
    # print resulting matrix
    # print(A, '\n')

    # solve the svd for the nullspace
    U, Sigma, Vt = np.linalg.svd(A)
    
    # return nullspace
    return Vt[-1]


if __name__ == '__main__':
    pass
