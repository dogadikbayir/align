import numpy as np
import glob

import qrotate as qr

""" Module adapted from https://github.com/ljmartin/align, original function definitions and code in repo. This is the forked version."""

import plotly
import plotly.graph_objs as go

def plotly_scatter(pointsets, sizes=[3,3,3,3,3]):
    """Plots some 3d point clouds. Usage: plotly_scatter([pointset1, pointset2], [size1, size2])"""
    data = []
    for p, s in zip(pointsets, sizes):
        a,b,c = p[:,0], p[:,1], p[:,2]
        trace = go.Scatter3d(x=a, y=b, z=c, mode='markers', marker={'size': s,'opacity': 0.8,})
        data.append(trace)
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    plot_figure = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(plot_figure)

def read_obj(fname):

    # Read from .obj mesh file into 3D point cloud
    with open(fname, 'r') as f:
        lines = f.readlines()
        pc = []
        faces = []
        for l in lines:
            l1 = l.split()
            if len(l1) != 0 and l1[0] == 'v':
                pc.append([float(l1[1]), float(l1[2] ) , float(l1[3])])

            if len(l1) != 0 and l1[0] == 'f':
                faces.append([int(l1[1]), int(l1[2]), int(l1[3])])

    return np.array(pc), np.array(faces)

def save_obj(fname, vertices, faces):

    # Save vertices, faces into .obj file
    with open(fname, 'w') as f:
        for v in vertices:
            f.write('v '+ ' '.join([str(vi) for vi in v]) + '\n')
        for fa in faces:
            f.write('f ' + ' '.join([str(fi) for fi in fa]) + '\n')

def compute_com(pts, masses=None):
    """ Computes the center of mass of an object. if masses != None, a mass vector needs to be provided"""
    if masses == None:
        masses = np.ones(pts.shape[0])
        masses /= masses.sum()

    return pts.T.dot(masses).astype('float64')

def calc_m_i(pcl):
    """ Computes the moment of inertia tensor """
    A = np.sum((pcl**2) * np.ones(pcl.shape[0])[:, None], 0).sum()
    B = (np.ones(pcl.shape[0]) * pcl.T).dot(pcl)

    eye = np.eye(3)
    return A * eye - B

def get_pmi(coords):
    """ Calculate principal moment of inertia"""
    momint = calc_m_i(coords)
    eigvals, eigvecs = np.linalg.eig(momint)

    # Sort
    indices = np.argsort(-eigvals) # sorting it returns the 'long' axis in index 0
    # Return transposed which is more intuitive format
    return eigvecs[:, indices].T

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotaxis(a, b):
    """Calculate the vector required in order to rotate `a` to align with `b`
    This is just the normal vector between the two vectors, normalized to unit length"""
    if np.allclose(a, b):
        return np.array([1, 0, 0])
    c = np.cross(a, b)
    return c / np.linalg.norm(c)

def align(vertices, axis=[0,1]):
    ''' Aligns the 3D shape's principal axii of inertia to x, y, z (given in axis)
        
        Since principal axii are orthogonal, we need to align the shape twice, once with axis[0] and once with axis[1], the remaining
        principal axis will be automatically aligned with one of the remaining x, y or z axii.

        Ex.: if axis = [0, 1], the principal axii of MOI will be aligned with x and y first, so the third principal axis of MOI will be au          tomatically aligned with z.

    '''
    # Translate shape to center of mass first
    center = vertices - compute_com(vertices)

    # Calculate MOI
    mi = calc_m_i(center)
    aligned = []
    for axidx in axis:
        
        # Create the axis vector
        axis_vector = np.zeros(3)
        axis_vector[axidx] = 1

        # get PMI axii
        pmi = get_pmi(center)

        # Get angle to that vector
        angle = angle_between(pmi[axidx], axis_vector)

        # Get axis around which to rotate
        ax = rotaxis(pmi[axidx], axis_vector)

        q = qr.from_axis_angle(ax, -angle)
        nq = qr.fast_normalise(q)
        rotmat = qr.quaternion_rotation_matrix(nq)

        aligned = center.dot(rotmat)

    return aligned
