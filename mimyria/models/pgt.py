import sys
import torch

from .base import PerAtomBaseNetwork
from mimyria.io import pgt_reshape

from e3nn.o3 import ReducedTensorProducts


class PGTNetwork(PerAtomBaseNetwork):
    def __init__(self, device, **kwargs):

        self.einsum_forward = 'ijkl, njkl->ni'
        self.einsum_backward = 'ijkl,...i->...jkl'

        rtp = ReducedTensorProducts('ijk=jik', i='1o', k='1o')
        self.irreps_out = rtp.irreps_out
        self.change_of_basis = rtp.change_of_basis

        # init
        super().__init__(device, **kwargs)

    ##############################
    # Overriding BaseNetwork
    ##############################

    def predict(self, data):
        # do the prediction
        y_preds = super().predict(data)

        # store the prediction in the atoms object
        for config, y_pred in zip(data, y_preds):
            if 'pgt' in config.arrays:
                del config.arrays['pgt']
            config.new_array('pgt', y_pred.cpu().detach().numpy())

    # computes the loss between the prediction and true data
    # y_pred is the prediction
    # y_ref is the true reference data
    def get_loss(self, y_pred, y_ref):
        return ((y_pred - y_ref)**2).mean()

    # returns the target property/ies for the given configuration stored in atoms
    def get_target(self, atoms):
        try:
            pgts = pgt_reshape(atoms.arrays['pgt'])
        except KeyError:
            print('ERROR: Polarizability Gradient Tensors (\"pgt\") must be contained in the ASE atoms object!', file=sys.stderr)
            raise

        # enforce symmetry!
        pgts[:] = 0.5 * (pgts + pgts.swapaxes(1, 2))

        return torch.from_numpy(pgts)
