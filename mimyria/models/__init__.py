import zipfile
import json

from mimyria.models.apt import APTNetwork
from mimyria.models.pgt import PGTNetwork


def model_from_target(target: str, **kwargs):
    models = {
            'apt': APTNetwork,
            'pgt': PGTNetwork
            }

    return models[target](**kwargs)


def load_model(device, model_fn):
    with zipfile.ZipFile(model_fn, 'r') as zf:
        info = json.load(zf.open('info.json'))

        if info['class_name'] == 'APTNetwork':
            target = 'apt'
        elif info['class_name'] == 'PGTNetwork':
            target = 'pgt'
        else:
            raise KeyError(f'Class {info["class_name"]} not found!')

        return model_from_target(target, device=device, load=model_fn)
