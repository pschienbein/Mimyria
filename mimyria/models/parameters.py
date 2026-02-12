
from enum import Enum, auto
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

import argparse
import torch


class EDataNormalization(Enum):
    NORM_NONE = auto()
    NORM_LAST = auto()
    NORM_IRREPS = auto()

    def __str__(self):
        return self.name


# helper for argparse
def enum_type(enum_cls):
    def convert(value):
        try:
            return enum_cls[value]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Invalid choice: {value}. Valid options: {', '.join(e.name for e in enum_cls)}"
                )
    return convert


@dataclass
class ModelParameters:
    atom_kinds: list
    radial_cutoff: float = 6.0
    num_radial_basis: int = 10
    # will be estimated from the training data
    num_neighbors: float = 0
    num_features: int = 50
    num_layers: int = 3
    lmax: int = 2
    normalization: EDataNormalization = EDataNormalization.NORM_LAST
    equal_num_features_per_channel: bool = True
    natural_parities_only: bool = False

    def to_dict(self):
        return {
            'atom_kinds': self.atom_kinds,
            'radial_cutoff': self.radial_cutoff,
            'num_radial_basis': self.num_radial_basis,
            'num_neighbors': self.num_neighbors,
            'num_features': self.num_features,
            'num_layers': self.num_layers,
            'lmax': self.lmax,
            'normalization': self.normalization.name,
            'equal_num_features_per_channel': self.equal_num_features_per_channel,
            'natural_parities_only': self.natural_parities_only
        }

    @classmethod
    def from_dict(cls, d):
        if isinstance(d['normalization'], str):
            d['normalization'] = EDataNormalization[d['normalization']]
        return cls(**d)

    def __str__(self):
        s = "Model Parameters\n"
        s += "================\n"

        fields2print = ['atom_kinds', 'radial_cutoff',
                        'num_neighbors', 'lmax',
                        'num_features', 'num_layers',
                        'normalization', 'equal_num_features_per_channel',
                        'natural_parities_only', 'num_radial_basis']

        for f in fields2print:
            if hasattr(self, f):
                s += f"  {f} = {getattr(self, f)!r}\n"
        return s


def get_default_apt(atom_kinds):
    return ModelParameters(atom_kinds=atom_kinds,
                           radial_cutoff=6.0,
                           num_radial_basis=10,
                           num_neighbors=0,
                           lmax=3,
                           num_features=20,
                           num_layers=3,
                           normalization=EDataNormalization.NORM_IRREPS,
                           equal_num_features_per_channel=False,
                           natural_parities_only=False)


def get_default_pgt(atom_kinds):
    return ModelParameters(atom_kinds=atom_kinds,
                           radial_cutoff=6.0,
                           num_radial_basis=10,
                           num_neighbors=0,
                           lmax=3,
                           num_features=40,
                           num_layers=3,
                           normalization=EDataNormalization.NORM_IRREPS,
                           equal_num_features_per_channel=False,
                           natural_parities_only=False)


def get_default(target, atom_kinds=[]):
    if target == 'pgt':
        return get_default_pgt(atom_kinds)
    else:
        return get_default_apt(atom_kinds)
