from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolAlign
from rdkit.Geometry import Point3D

class Mol3D(object):
    """
    Basically a wrapper for some  rdkit functions:
    - make a molecule
    - embed multiple conformers in 3d space
    - calculate Gasteiger charges for Sinkhorn calculation
    - also calculate MMFF forcefield parameters for traditional 3d alignment calculation.
    """
    
    def __init__(self, smiles):
        self.mol = Chem.MolFromSmiles(smiles)
        self.pos = None
        self.charges = None
        
        self.twoD = Chem.MolFromSmiles(smiles) #keep a 2d rep for visualization
        
    def embed3d(self, nconf=100, hydrogens=False):
        """Add hydrogens, generate a 3d conformer"""  
        
        self.mol = Chem.AddHs(self.mol)
        conformer_idx = AllChem.EmbedMultipleConfs(self.mol, numConfs=nconf)
        if not hydrogens:
            self.mol = Chem.RemoveHs(self.mol)
        
        self.zeroth_conf = self.mol.GetConformer(0)
        
        
    def get_pos(self, confid):
        """Fetch the x,y,z coords of a conformer"""
        
        return self.mol.GetConformer(confid).GetPositions()

    def set_pos(self, positions):
        """Set the base (zeroth) conformer xyz coords"""
        
        for i in range(self.mol.GetNumAtoms()):
            x,y,z = positions[i]
            self.zeroth_conf.SetAtomPosition(i,Point3D(x,y,z))
            
    def gen_mmff(self):
        """Calculate MMFF forcefield properties"""
        self.mmffprops = AllChem.MMFFGetMoleculeProperties(self.mol)

    def gen_charges(self):
        """Calculate Gasteiger charges"""
        Chem.rdPartialCharges.ComputeGasteigerCharges(self.mol, throwOnParamFailure=True)
        charges = [float(self.mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(self.mol.GetNumAtoms())]
        self.charges = np.array(charges)
            
    def get_descriptor(self, confid):
        """Get 2d charge-distance descriptor."""
        if self.charges is None:
            self.gen_charges()
        
        #generate a new set of positions:
        pos = self.get_pos(confid)
        
        return np.vstack([pdist(pos), pdist(self.charges[:,None])])
