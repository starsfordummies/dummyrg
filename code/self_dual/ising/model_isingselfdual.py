from tenpy.models.lattice import Site, Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
#from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite
import numpy as np

class TFISDModel(CouplingMPOModel):
    r"""Self-dual Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = SpinHalfSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        p = np.asarray(model_params.get('p', 1.))
        lam = np.asarray(model_params.get('lambda', 1.))

        # First the two "standard" transverse Ising pieces (here J=1)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-lam, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-1., u1, 'Sigmax', u2, 'Sigmax', dx)

        # Then the two new pieces ~p 
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(p*lam, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(p, u1, 'Sigmax', u2, 'Sigmax', dx)
        # done

# We can't build trivially the Hamiltonian as NearestNeighbor with the next-nearest interaction
#class TFISDChain(TFISDModel, NearestNeighborModel):

# We can on the other hand build the MPO version 
# TODO: check if I'm not missing something here 
class TFISDChain(TFISDModel, MPOModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

