#!/usr/bin/env python
import argparse
import torch


def average_models(model_files):
    avg_model = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['state_dict']

        if i == 0:
            avg_model = model_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

    final = {"state_dict": avg_model}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("models", metavar='MODELS', nargs="+",
                        help="List of models")
    parser.add_argument("-output", "-o", default="averaged_model.pt",
                        help="Output file")
    opt = parser.parse_args()

    final = average_models(opt.models)
    torch.save(final, opt.output)


if __name__ == "__main__":
    main()
