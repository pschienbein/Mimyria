#!/bin/env python3

import argparse

from mimyria.models import load_model


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Loads a given module an prints the model parameters')
    parser.add_argument('--model', type=str, default='model.torch')
    args = parser.parse_args(argv)

    ###############################################

    mdl = load_model('cpu', args.model)
    print(mdl.model_parameters)


if __name__ == "__main__":
    main()
