import io
import sys
from typing import Dict, Union
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict
import zipfile
import json

import torch

import numpy as np
from ase import Atoms

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.math import soft_one_hot_linspace

from .parameters import ModelParameters, EDataNormalization

# ensure double precision
torch.set_default_dtype(torch.float64)


# class ModifiedSimpleNetwork(SimpleNetwork):
class ModifiedSimpleNetwork(torch.nn.Module):
    def __init__(self, parent, model_parameters):
        # print all parameters to stdout
        print(model_parameters)

        # initialize torch.nn.Module
        super().__init__()

        self.max_radius = model_parameters.radial_cutoff
        self.number_of_basis = model_parameters.num_radial_basis

        lmax = self.lmax = model_parameters.lmax
        num_neighbors = model_parameters.num_neighbors
        irreps_in = parent.irreps_in
        irreps_out = parent.irreps_out
        layers = model_parameters.num_layers

        Nf = model_parameters.num_features
        mul = dict()
        for l in range(lmax + 1):
            if model_parameters.equal_num_features_per_channel:
                mul[l] = Nf
            else:
                mul[l] = max(4, Nf // (l + 1))

        if model_parameters.natural_parities_only:
            irreps_node_hidden = Irreps([
                (mul[l], (l, 1 if l % 2 == 0 else -1))
                for l in range(0, lmax + 1)
                ])
        else:
            irreps_node_hidden = Irreps([(mul[l], (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_in] + layers * [irreps_node_hidden] + [irreps_out],
            irreps_node_attr="0e",
            irreps_edge_attr=Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

        print('EDGE l:', self.mp.irreps_edge_attr)
        print('HIDDEN irreps:', irreps_node_hidden)
        # print('DEPTH', len(self.mp.irreps_node_sequence)-2)
        print('PARAMETERS', sum(p.numel() for p in self.mp.parameters()), flush=True)

        pass

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]

        edge_vec = data['edge_vec']

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        return node_outputs


class BaseNetwork(ABC):
    def __init__(self, device, **kwargs):
        self.device = device
        self.bRestart = False

        # setup a new model
        if 'model_parameters' in kwargs:
            self.model_parameters = kwargs['model_parameters']
            self.irreps_in = str(len(self.model_parameters.atom_kinds)) + "x0e"
            self.net = None
            self.optimizer = None
            self.scheduler = None
            self.train_data_indices = []

        # load an existing model
        elif 'load' in kwargs:
            try:
                sFn = kwargs['load']
                self.load(sFn)
                self.bRestart = True

            except Exception as e:
                print(f"Data file \"{sFn}\" could not be loaded!", file=sys.stderr)
                print(f"the exception is: \"{e}\".", file=sys.stderr)
                print(type(e).__name__)
                exit(1)

        ######################################
        # initializations independent of load or init
        self.change_of_basis_on_gpu = self.change_of_basis.to(self.device)

        # determine irrep blocks
        blocks = []
        offset = 0
        for mul, ir in self.irreps_out:
            mdim = 2 * ir.l + 1
            for copy in range(mul):
                sl = slice(offset, offset + mdim)
                blocks.append({
                    'l': ir.l,
                    'p': ir.p,
                    'copy': copy,
                    'mdim': mdim,
                    'sl': sl,
                    'name': f"{ir.l}{'e' if ir.p == 1 else 'o'}_{copy}"
                    })
                offset += mdim
        self.irrep_blocks = blocks

        self.tmp_full_graph_size = 0

    def create_graph_from_frame(self, atoms):
        cutoff2 = pow(self.model_parameters.radial_cutoff, 2.0)

        N = len(atoms)
        size = N*N

        if self.tmp_full_graph_size != N:
            t1 = []
            t2 = []

            for i in range(N):
                for j in range(N):
                    t1.append(i)
                    t2.append(j)

            self.tmp_full_graph_edge_src = torch.tensor(np.array(t1), dtype=torch.long, device=self.device)
            self.tmp_full_graph_edge_dst = torch.tensor(np.array(t2), dtype=torch.long, device=self.device)

            self.tmp_full_graph_size = N

        positions = torch.tensor(atoms.get_positions(), device=self.device)

        vd = positions[self.tmp_full_graph_edge_dst[:]] - positions[self.tmp_full_graph_edge_src[:]]

        # APPLY PBC
        lattice_vectors = torch.tensor(atoms.cell.array, device=self.device)
        lattice_vectors_inverse = torch.inverse(lattice_vectors)

        frac = torch.einsum('ij,nj->ni', lattice_vectors_inverse, vd)
        shift = torch.round(frac)
        vd -= torch.einsum('ij,nj->ni', lattice_vectors, shift)
        d = torch.einsum('ni,ni->n', vd, vd)

        mask = d < cutoff2

        edge_src = torch.masked_select(self.tmp_full_graph_edge_src, mask)
        edge_dst = torch.masked_select(self.tmp_full_graph_edge_dst, mask)
        edge_vec = torch.masked_select(vd, mask.unsqueeze(-1).expand(vd.size())).reshape((len(edge_src), 3))

        return edge_src, edge_dst, edge_vec

    def frame2Data(self, config):
        N = len(config)

        x_index = torch.tensor([self.model_parameters.atom_kinds.index(d) for d in config.get_chemical_symbols()])
        x = torch.zeros(N, len(self.model_parameters.atom_kinds))
        x[range(N), x_index] = 1.0

        edge_src, edge_dst, edge_vec = self.create_graph_from_frame(config)

        return {'pos': torch.tensor(config.get_positions()).to(self.device),
                'x': x.to(self.device),
                'species_idx': x.to(self.device).argmax(1).to(torch.long),
                'edge_index': torch.stack([edge_src, edge_dst], dim=0),
                'edge_vec': edge_vec}

    def frame2TrainData(self, config):
        pt = self.frame2Data(config)

        y = self.get_target(config).to(self.device)
        return Data(
                pos=pt['pos'],
                x=pt['x'],
                edge_index=pt['edge_index'],
                edge_vec=pt['edge_vec'],
                symbols=config.get_chemical_symbols(),
                species_idx=pt['species_idx'],
                y_internal=self.y_to_internal(y, config),
                y=y)

    #
    def frame2PredictData(self, config):
        pt = self.frame2Data(config)
        return Data(
                pos=pt['pos'],
                x=pt['x'],
                species_idx=pt['species_idx'],
                edge_index=pt['edge_index'],
                edge_vec=pt['edge_vec'])

    def predict(self, data):
        with torch.no_grad():
            tdata = []
            for config in data:
                tdata.append(self.frame2PredictData(config))

            dataloader = DataLoader(tdata, len(data))
            retdata = []
            for ex in dataloader:
                y_internal_pred = self.net({'pos': ex.pos, 'x': ex.x, 'edge_index': ex.edge_index, 'edge_vec': ex.edge_vec, 'batch': ex.batch})
                # NOTE This assumes that the symbols do not change within the batch
                y_pred = self.y_from_internal(y_internal_pred, config)

                atom_splits = ex.ptr
                batch_pred = [
                        y_pred[atom_splits[i]:atom_splits[i+1]]
                        for i in range(atom_splits.size(0) - 1)
                        ]

                retdata.extend(batch_pred)

            return retdata

    def train(self,
              data,
              num_epochs=1000,
              learning_curve_fout='learning-curve.dat',
              batch_size=1,
              initial_learning_rate=0.01,
              learning_curve_per_irrep_fout=None):

        # log
        if isinstance(learning_curve_fout, io.IOBase):
            # use the opened stream
            ofslog = learning_curve_fout
        else:
            # interpret as file name
            ofslog = open(learning_curve_fout, 'w')

        # print training characteristics
        print('# Training settings:')
        print('# ==================')
        print(f'# Batch Size: {batch_size}')
        print(f'# Initial Learning rate: {initial_learning_rate}')

        # logging per irrep
        bLCPerIrrep = False
        if learning_curve_per_irrep_fout is not None:
            bLCPerIrrep = True
            if isinstance(learning_curve_per_irrep_fout, io.IOBase):
                # use the provided stream
                ofslog_irrep = learning_curve_per_irrep_fout
            else:
                ofslog_irrep = open(learning_curve_per_irrep_fout, 'w')

        # Print the model parameters to the LC output to preserve them
        # ensure that each line is a comment ('#')
        param_str = str(self.model_parameters)
        ofslog.write('\n'.join("# " + line for line in param_str.splitlines()))
        ofslog.write('\n')

        # train / test split
        if len(self.train_data_indices) == 0:
            frac = 0.9
            ndata = len(data)

            self.train_data_indices = np.random.choice(range(ndata), int(frac*ndata), replace=False)
            self.train_data_indices = self.train_data_indices.astype(int).tolist()
            self.test_data_indices = list(set(range(ndata)) - set(self.train_data_indices))

            if len(self.train_data_indices) == 0:
                self.train_data_indices = self.test_data_indices
                self.test_data_indices = []

            ofslog.write('# training indices: ')
            for i in self.train_data_indices:
                ofslog.write(f'{i} ')
            ofslog.write('\n# test indices: ')
            for i in self.test_data_indices:
                ofslog.write(f'{i} ')
            ofslog.write('\n')

        # determine normalization:
        # (only training data)
        self.determine_normalization(
                [data[i] for i in self.train_data_indices]
                )

        # prepare training
        train_data = []
        for idx in self.train_data_indices:
            train_data.append(self.frame2TrainData(data[idx]))

        test_data = []
        for idx in self.test_data_indices:
            test_data.append(self.frame2TrainData(data[idx]))

        # estimate the number of neighbors, if not set
        if self.model_parameters.num_neighbors == 0:
            N = []
            for d in itertools.chain(train_data, test_data):
                Natoms = len(d.symbols)
                Nedges = len(d.edge_index[0])
                N.append(float(Nedges) / Natoms)

            self.model_parameters.num_neighbors = int(round(np.mean(N)))

        # if this is not a restart, create the NN, optimizer and scheduler
        if not self.bRestart:
            self.net = ModifiedSimpleNetwork(self, self.model_parameters)
            self.net.to(self.device)
            self.optimizer = Adam(self.net.parameters(), lr=initial_learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer)

        # 
        num_params = sum(p.numel() for p in self.net.mp.parameters())
        ofslog.write(f'# number of parameters: {num_params}\n')

        # Training loop
        ofslog.write('# Epoch | Train Loss (internal) | Test Loss (internal) | Train Loss | Test Loss | Learning Rate\n')

        # learning curve per irrep:
        if bLCPerIrrep:
            ofslog_irrep.write('# All loss reported in internal space')
            ofslog_irrep.write('# Epoch | ')
            for stage in ['Train', 'Test']:
                for blk in self.irrep_blocks:
                    ofslog_irrep.write(f'{stage} Loss {blk["name"]} | ')

            ofslog_irrep.write('\n')

        # Training loop
        dataloaders = {
                "train": DataLoader(train_data, batch_size, shuffle=True),
                "test": DataLoader(test_data, batch_size)
                }

        epoch = 0
        while epoch < num_epochs:
            for phase in ['train', 'test']:
                if phase == 'test' and len(self.test_data_indices) == 0:
                    test_loss = 0
                    test_loss_internal = 0
                    continue

                if phase == 'test':
                    self.net.eval()
                    torch.set_grad_enabled(False)
                else:
                    self.net.train()
                    torch.set_grad_enabled(True)

                running_loss_internal = 0.0
                running_loss = 0.0
                norm = 0.0
                per_species = defaultdict(list)

                if bLCPerIrrep:
                    irrep_epoch_sums = {i: 0.0 for i, _ in enumerate(self.irrep_blocks)}
                    irrep_epoch_counts = {i: 0 for i, _ in enumerate(self.irrep_blocks)}

                for ex in dataloaders[phase]:
                    ex = ex.to(self.device)

                    y_internal_pred = self.net({
                        'pos': ex.pos,
                        'x': ex.x,
                        'edge_index': ex.edge_index,
                        'edge_vec': ex.edge_vec,
                        'batch': ex.batch
                        })

                    # this is a micro loss
                    loss_internal = self.get_loss(y_internal_pred, ex.y_internal)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss_internal.backward()
                        self.optimizer.step()
                    running_loss_internal += loss_internal.detach() * ex.x.size(0)

                    # Per Irrep learning curve
                    if bLCPerIrrep:
                        for i, blk in enumerate(self.irrep_blocks):
                            sl = blk['sl']
                            err = ((y_internal_pred[:, sl] - ex.y_internal[:, sl]) ** 2).mean(dim=1).sqrt()
                            irrep_epoch_sums[i] += err.sum().item()
                            irrep_epoch_counts[i] += err.numel()

                    # the rmse is reported in internal target representation
                    symbols = list(itertools.chain.from_iterable(ex.symbols))
                    y_pred = self.y_from_internal(y_internal_pred, symbols)
                    loss = self.get_loss(y_pred, ex.y)
                    running_loss += loss.detach() * ex.x.size(0)

                    # species RMSE
                    for i, sym in enumerate(self.model_parameters.atom_kinds):
                        mask = (ex.species_idx == i)
                        if mask.any():
                            per_species[sym].append(self.get_loss(y_pred[mask], ex.y[mask]))

                    norm += ex.x.size(0)

                # epoch_loss = running_loss / norm
                epoch_loss_internal = running_loss_internal / norm

                for sym in per_species:
                    per_species[sym] = torch.sqrt(torch.stack(per_species[sym]).mean())
                macro_rmse = torch.stack(list(per_species.values())).mean().item()

                epoch_loss_irrep_internal = None
                if bLCPerIrrep:
                    epoch_loss_irrep_internal = {
                            i: irrep_epoch_sums[i] / irrep_epoch_counts[i]
                            for i in irrep_epoch_counts
                        }

                if phase == 'test':
                    test_macro_rmse = macro_rmse
                    test_loss_internal = epoch_loss_internal.item()
                    test_loss_irrep_internal = epoch_loss_irrep_internal

                    self.scheduler.step(macro_rmse)
                else:
                    train_macro_rmse = macro_rmse
                    train_loss_internal = epoch_loss_internal.item()
                    train_loss_irrep_internal = epoch_loss_irrep_internal

                # End of training loop for this epoch
                ################

            # append to the learning curve
            ofslog.write('%d %e %e %e %e %e\n'%(
                epoch,
                np.sqrt(train_loss_internal),
                np.sqrt(test_loss_internal),
                train_macro_rmse,
                test_macro_rmse,
                self.optimizer.param_groups[0]['lr'])
                )

            if bLCPerIrrep:
                ofslog_irrep.write(f'{epoch} ')
                for i in range(len(self.irrep_blocks)):
                    ofslog_irrep.write(f'{train_loss_irrep_internal[i]} ')
                for i in range(len(self.irrep_blocks)):
                    ofslog_irrep.write(f'{test_loss_irrep_internal[i]} ')
                ofslog_irrep.write('\n')
                ofslog_irrep.flush()

            ofslog.flush()

            epoch += 1

            # early stopping, if learning rate is small enough
            if self.optimizer.param_groups[0]['lr'] < 1e-7:
                break

        self.last_epoch = epoch

    ########################
    # I/O
    ########################

    def save(self, fn):
        def put_torch(zf, data, int_fn):
            buffer = io.BytesIO()
            torch.save(data, buffer)
            buffer.seek(0)
            zf.writestr(int_fn, buffer.getvalue())

        with zipfile.ZipFile(fn, mode='w') as zf:
            put_torch(zf, self.net.state_dict(), 'net.torch')
            put_torch(zf, self.optimizer.state_dict(), 'opt.torch')
            put_torch(zf, self.scheduler.state_dict(), 'scheduler.torch')
            put_torch(zf, self.state_dict(), 'state.torch')

            info = self.fetch_info_dict()
            zf.writestr('info.json', json.dumps(info))

            params = self.model_parameters.to_dict()
            zf.writestr('params.json', json.dumps(params))

        pass

    # Note: This load function keeps everthing separate,
    #       No need to load the full torch array (or even loading torch at all)
    #       To get model information or parameters
    def load(self, model_fn):
        def get_torch(zf, int_fn):
            return torch.load(io.BytesIO(zf.read(int_fn)), map_location=self.device)

        with zipfile.ZipFile(model_fn, 'r') as zf:
            params = json.load(zf.open('params.json'))
            self.model_parameters = ModelParameters.from_dict(params)
            self.irreps_in = str(len(self.model_parameters.atom_kinds)) + "x0e"

            self.net = ModifiedSimpleNetwork(self, self.model_parameters)
            self.net.load_state_dict(get_torch(zf, 'net.torch'))
            self.net.to(self.device)

            self.optimizer = Adam(self.net.parameters(), lr=0.01)
            self.optimizer.load_state_dict(get_torch(zf, 'opt.torch'))

            self.scheduler = ReduceLROnPlateau(self.optimizer)
            self.scheduler.load_state_dict(get_torch(zf, 'scheduler.torch'))

            # load extended model information
            self.load_state_dict(get_torch(zf, 'state.torch'))
            self.load_info_dict(json.load(zf.open('info.json')))

    def fetch_info_dict(self):
        return {
            'class_name': self.__class__.__name__,
            'train_data_indices': self.train_data_indices,
            'test_data_indices': self.test_data_indices,
            'last_epoch': self.last_epoch,
        }

    def load_info_dict(self, d):
        self.train_data_indices = d['train_data_indices']
        self.test_data_indices = d['test_data_indices']
        self.last_epoch = d['last_epoch']

    def state_dict(self):
        return {
            'change_of_basis': self.change_of_basis
        }

    def load_state_dict(self, sd):
        self.change_of_basis = sd['change_of_basis']

    ##############################
    # Purely virtual functions
    ##############################

    # returns the target property/ies for the given configuration stored in atoms
    @abstractmethod
    def get_target(self, atoms):
        pass

    # computes the loss between the prediction and true data
    # y_pred is the prediction
    # y_ref is the true reference data
    # NOTE both are in the internal basis and normalized
    @abstractmethod
    def get_loss(self, y_pred, y_ref):
        pass

    @abstractmethod
    def determine_normalization(self, configs):
        pass

    # takes the target tensor y and transforms it into a one dimensional array which is then trained; 
    # process might require a change of basis and/or normalization
    @abstractmethod
    def y_to_internal(self, y, config):
        pass

    # takes an internal target tensor y and transforms it back into the original representation; 
    # process might require a change of basis and/or normalization
    # config can be an Atoms object, then it is just one single configuration
    # can also be a list of atom symbols, that happens during the training process, if many frames are processed in one batch
    def y_from_internal(self, y_internal, config : Union[list, Atoms]):
        pass

#################################################
# Base network for training a per-atom quantity
#################################################


class PerAtomBaseNetwork(BaseNetwork):
    def __init__(self, device, **kwargs):
        # normalization arrays
        self.normalize_mean = None
        self.normalize_std = None
        self.norm_work_symbols = ""

        super().__init__(device, **kwargs)

    def get_norm_working_arrays(self, atoms, num_atoms_in_batch):
        if self.norm_work_symbols != str(atoms.symbols):
            # update the working arrays
            self.norm_work_symbols = str(atoms.symbols)
            self.norm_work_mean = torch.stack([self.normalize_mean[d] for d in atoms.get_chemical_symbols()])
            self.norm_work_std = torch.stack([self.normalize_std[d] for d in atoms.get_chemical_symbols()])
            self.norm_work_batches = 1

        atoms_per_batch = self.norm_work_mean.shape[0]
        repeats = num_atoms_in_batch // atoms_per_batch

        if repeats == 1:
            return self.norm_work_mean, self.norm_work_std
        else:
            # batched!
            if self.norm_work_batches != repeats:
                self.norm_work_mean_batched = self.norm_work_mean.repeat((repeats, 1))
                self.norm_work_std_batched = self.norm_work_std.repeat((repeats, 1))
                self.norm_work_batches = repeats

            return self.norm_work_mean_batched, self.norm_work_std_batched

    ##################################
    # Overriding BaseNetwork
    ##################################

    def determine_normalization(self, configs):

        if self.normalize_mean is None:
            # if no normalization requested, exit
            if self.model_parameters.normalization == EDataNormalization.NORM_NONE:
                return

            atom_data = defaultdict(list)
            self.normalize_mean = dict()
            self.normalize_std = dict()

            # for iFrame in self.train_data_indices:
            for iFrame in range(len(configs)):
                y = self.get_target(configs[iFrame])
                y = torch.einsum(self.einsum_forward, self.change_of_basis, y)

                # sort depending on the atom type
                for iAtom, atom in enumerate(configs[iFrame]):
                    atom_data[atom.symbol].append(y[iAtom])

            # proceed depending on the chosen normalization
            if self.model_parameters.normalization == EDataNormalization.NORM_LAST:
                for symb in atom_data:
                    stack = torch.stack(atom_data[symb])
                    self.normalize_mean[symb] = torch.mean(stack, dim=0).to(self.device)
                    self.normalize_std[symb] = torch.std(stack, dim=0, correction=0).to(self.device)

            elif self.model_parameters.normalization == EDataNormalization.NORM_IRREPS:
                eps = 1e-12
                for symb, items in atom_data.items():
                    stack = torch.stack(items, dim=0).to(self.device)

                    means = torch.zeros(stack.shape[1], device=self.device)
                    stds = torch.zeros_like(means)

                    for blk in self.irrep_blocks:
                        sl = blk['sl']
                        l = blk['l']
                        Y = stack[:, sl]

                        if l == 0:
                            mu = Y.mean(dim=0)
                            Yc = Y - mu
                            sigma = (Yc.pow(2).mean(dim=0)).sqrt()
                            means[sl] = mu
                            stds[sl] = torch.clamp(sigma, min=eps)

                        else:
                            rms_scalar = (Y.pow(2).sum(dim=1) + eps).sqrt().mean()
                            means[sl] = 0.0
                            stds[sl] = torch.clamp(rms_scalar, min=eps).expand_as(means[sl])

                    self.normalize_mean[symb] = means
                    self.normalize_std[symb] = stds

    def y_to_internal(self, y, config):
        norm_work_mean, norm_work_std = self.get_norm_working_arrays(config, len(config))

        if self.model_parameters.normalization == EDataNormalization.NORM_NONE:
            # NO NORM:
            print('debug check: NONORM', flush=True)
            y_internal = torch.einsum(self.einsum_forward, self.change_of_basis_on_gpu, y)

        else:
            y_base = torch.einsum(self.einsum_forward, self.change_of_basis_on_gpu, y)
            y_internal = (y_base - norm_work_mean) / norm_work_std

        return y_internal

    def y_from_internal(self, y_internal, config):

        if isinstance(config, Atoms):
            work_mean, work_std = self.get_norm_working_arrays(config, len(y_internal))
        else:
            symbols = config
            work_mean = torch.stack([self.normalize_mean[d] for d in symbols])
            work_std = torch.stack([self.normalize_std[d] for d in symbols])

        if self.model_parameters.normalization == EDataNormalization.NORM_NONE:
            # NO NORM:
            y_pred = torch.einsum(self.einsum_backward, self.change_of_basis_on_gpu, y_internal)

        else:
            y_denorm = y_internal.detach() * work_std + work_mean
            y_pred = torch.einsum(self.einsum_backward, self.change_of_basis_on_gpu, y_denorm)

        return y_pred

    ########################
    # I/O
    ########################

    def fetch_info_dict(self):
        sd = super().fetch_info_dict()

        sd.update({
        })

        return sd

    def state_dict(self):
        sd = super().state_dict()

        sd.update({
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std
            })

        return sd

    def load_state_dict(self, sd):
        super().load_state_dict(sd)

        self.normalize_mean = sd['normalize_mean']
        self.normalize_std = sd['normalize_std']

#################################################
# Base network for training a global quantity
#################################################


class GlobalBaseNetwork(BaseNetwork):
    def __init__(self, device, **kwargs):
        # normalization arrays
        self.normalize_mean = None
        self.normalize_std = None
        self.norm_work_symbols = ""

        super().__init__(device, **kwargs)

    def get_norm_working_arrays(self, atoms):
        if self.norm_work_symbols != str(atoms.symbols):
            # update the working arrays
            self.norm_work_symbols = str(atoms.symbols)
            self.norm_work_mean = torch.stack([self.normalize_mean[d] for d in atoms.get_chemical_symbols()])
            self.norm_work_std = torch.stack([self.normalize_std[d] for d in atoms.get_chemical_symbols()])

        return self.norm_work_mean, self.norm_work_std

    ##################################
    # Overriding BaseNetwork
    ##################################

    def determine_normalization(self, configs):
        if self.normalize_mean is None:
            ys = []
            for config in configs:
                ys.append(self.get_target(config))

            ys = torch.stack(ys)
            self.normalize_mean = torch.mean(ys, dim=0)
            self.normalize_std = torch.std(ys, dim=0)

            print(self.normalize_mean)
            print(self.normalize_std)

    def y_to_internal(self, y, config):
        y_base = y
        y_internal = y_base
        return y_internal

    def y_from_internal(self, y_internal, config):
        y_denorm = y_internal.detach()
        y_pred = y_denorm
        return y_pred

    ########################
    # I/O
    ########################

    def fetch_info_dict(self):
        sd = super().fetch_info_dict()

        sd.update({
        })

        return sd

    def state_dict(self):
        sd = super().state_dict()

        sd.update({
            })

        return sd

    def load_state_dict(self, sd):
        super().load_state_dict(sd)

        self.normalize_mean = sd['normalize_mean']
        self.normalize_std = sd['normalize_std']
