import sys
import torch

from .base import PerAtomBaseNetwork
from mimyria.io import apt_reshape

from e3nn.o3 import ReducedTensorProducts


class APTNetwork(PerAtomBaseNetwork):
    def __init__(self, device, **kwargs):

        self.einsum_forward = 'ijk, njk->ni'
        self.einsum_backward = 'ijk,...i->...jk'

        rtp = ReducedTensorProducts('ij', i='1o', j='1o')
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
            if 'apt' in config.arrays:
                del config.arrays['apt']
            config.new_array('apt', y_pred.cpu().detach().numpy())

    # computes the loss between the prediction and true data
    # y_pred is the prediction
    # y_ref is the true reference data
    def get_loss(self, y_pred, y_ref):
        return ((y_pred - y_ref)**2).mean()

    # returns the target property/ies for the given configuration stored in atoms
    def get_target(self, atoms):
        try:
            apts = apt_reshape(atoms.arrays['apt'])

        except KeyError:
            print('ERROR: apts must be contained in the ASE atoms object!', file=sys.stderr)
            raise

        return torch.from_numpy(apts)
