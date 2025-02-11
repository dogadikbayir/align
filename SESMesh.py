import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf

class SESMesh(object):
    def __init__(self, proberad=1.4):
        self.PROBERAD = proberad
    
    def circle_points(self, num_pts, radius=1):
        """Sample points spread out around a 2D unit circle"""
        t = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        
        self.template_points = np.c_[x, y]
    
    def golden_spiral(self, num_pts, radius=1):
        """Sample points spread out around a 3D unit sphere surface
        See stackoverflow post: ###"""
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
        points = np.vstack([x,y,z]).T
        
        self.template_points = points*radius
    
    def generate_points(self, positions, radii):
        """This function samples points on the surface of the atoms (blue, surface point)
        as well as on the extended surface (orange).
        
        The logic is that you first repeat the spherical surface points by the number of atoms.
        Then multiply them all by the radii. Since they're all still centred on zero, they just
        assume the shapes of the atoms now. Then, translate them all by the atom coordinates.
        """
        self.positions = positions
        self.radii = radii
        
        n = self.template_points.shape[0]
        ##repeat the atomic radii by 'n' to keep track of how far each blue point should be from 
        ##its parent atom.
        self.all_radii = np.repeat(self.radii, n)

        ##repeat the surface points, and then multiply by atomic radii
        sp = np.tile(self.template_points, (self.positions.shape[0],1)) * np.repeat(self.radii, n)[:,None]
        ##now translate by positions
        self.sp = sp+np.repeat(self.positions, n,axis=0)

        ##similarly, repeat the surface points and multiply by atomic radii + PROBERAD
        ep = np.tile(self.template_points,(len(self.positions),1)) *np.repeat(self.radii+self.PROBERAD, n)[:,None]
        #now transform by positions
        self.ep = ep+np.repeat(self.positions, n,axis=0)
        
        
    def remove_bad_surface_points(self):
        """
        This calculates the distance between surface points and the atom centres. 
        If a point-atom distance is less than an atom radius, then the surface point
        must be within another atom. Delete those points by masking.
        """
        #remove any blue surface point that is inside another atom 
        dmat = cdist(self.sp, self.positions)
        inside = ((dmat-self.radii)+0.00001).min(1) < 0

        #apply the mask:
        self.sp = self.sp[~inside]
        
    def remove_bad_extended_points(self):
        """
        Same as `remove_bad_surface_points` but for extended points.
        This calculates the distance between extended points and the atom centres. 
        If a point-atom distance is less than an atom radius plus the probe radius, 
        then the surface point must be too close to the atom surface. 
        Delete those points by masking.
        """
        #remove any orange extended points that you could not 
        #centre a probe atom on.
        dmat = cdist(self.ep, self.positions) - self.radii
        too_close = dmat.min(1)<(self.PROBERAD-0.000001)

        #apply the mask
        self.ep = self.ep[~too_close]
        
    def translate_reentrant_surfaces(self):
        """
        This identifies any surface point that is _more_ than one PROBERAD
        away from its closest extended point. That means this surface point must be
        hidden within a reentrant surface. 
        It then finds the vector pointing to the nearest extended point, and
        translates each hidden surface point to be exactly one PROBERAD away. 
        """
        dmat = cdist(self.ep, self.sp)
        
        surface_idx = dmat.min(0)>(self.PROBERAD+0.0001)
        extended_idx = dmat[:,surface_idx].argmin(0)

        vec = self.ep[extended_idx] - self.sp[surface_idx]

        scaling = 1 - (self.PROBERAD / np.linalg.norm(vec,axis=1))
        self.sp[surface_idx] +=(scaling[:,None]*vec)
        
    def sample_sdf(self):
        """
        To generate a signed distance field, we can take some sampled distances 
        and interpolate a grid from that. Some convenient sample points are the existing
        surface points and extended points. This 
        """
        surface_distances = np.zeros(self.sp.shape[0])
        extended_distances = np.ones(self.ep.shape[0])*self.PROBERAD
        
        self.sampled_coords = np.vstack([self.sp,self.ep, self.positions])
        self.sampled_sdf = np.hstack([surface_distances,extended_distances, -self.radii])


        
    def gen_grid(self, gridsize=0.5, buf=None):
        self.gridsize = gridsize
        if buf is None:
            buf = self.PROBERAD 
        
        min_coords = self.sampled_coords.min(0)-buf
        max_coords = self.sampled_coords.max(0)+buf
        self.max_coords = max_coords + (gridsize - (max_coords)%gridsize)
        self.min_coords = min_coords - (gridsize + (min_coords)%gridsize)
        
        self.bins = list()
        
        for dim in range(self.min_coords.shape[0]):
            print(dim)
            grid = np.linspace( self.min_coords[dim], 
                               self.max_coords[dim], 
                               int((self.max_coords[dim]-self.min_coords[dim])/gridsize+1)
                              )
            self.bins.append(grid)
            
        self.mgrid = np.meshgrid(*self.bins)
        self.grid = [i.ravel() for i in self.mgrid]
        
    def interp(self):
        r = Rbf(*self.sampled_coords.T, self.sampled_sdf)
        self.sdf = r(*self.grid).reshape(self.mgrid[0].shape)
