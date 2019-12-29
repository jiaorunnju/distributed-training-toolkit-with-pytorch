#!/usr/bin/env python

import os
import torch
import argparse

parser = argparse.ArgumentParser(description="Release model from checkpoint")
parser.add_argument("model", metavar='MODEL', help="checkpoint file")
parser.add_argument('-output', default='released_model.pt', type=str)

args = parser.parse_args()

assert os.path.isfile(args.model), "checkpoint file does not exist!"

ckpt = torch.load(args.model)
ckpt['optimizer'] = None
torch.save(ckpt, args.output)
