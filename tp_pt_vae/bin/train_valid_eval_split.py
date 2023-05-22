# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import glob
import os
import pathlib
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, help="data directory containing npy", required=True)
    parser.add_argument("-o1", "--output_train", type=str, help="train file list file", required=True)
    parser.add_argument("-o2", "--output_valid", type=str, help="validation file list file", required=True)
    parser.add_argument("-o3", "--output_eval", type=str, help="evaluation file list file", required=True)
    parser.add_argument("-r1", "--rate_train", type=float, help="train data rate", default=0.9)
    parser.add_argument("-r2", "--rate_valid", type=float, help="train data rate", default=0.05)
    parser.add_argument("-r3", "--rate_eval", type=float, help="train data rate", default=0.05)
    args = parser.parse_args()
    data_root_dir = args.data

    npys = glob.glob(os.path.join(data_root_dir, "*.npy"))
    if len(npys) == 0:
        print("No training data found")
        exit()
    random.shuffle(npys)

    # create parent directory if not exists
    if not os.path.exists(pathlib.Path(args.output_train).parent):
        os.makedirs(pathlib.Path(args.output_train).parent)
    if not os.path.exists(pathlib.Path(args.output_valid).parent):
        os.makedirs(pathlib.Path(args.output_valid).parent)


    p1 = int(len(npys) * args.rate_train)
    p2 = p1 + int(len(npys) * args.rate_valid)
    trains = npys[:p1]
    valids = npys[p1:p2]
    evals = npys[p2:]

    with open(args.output_train, "w") as f:
        for file in trains:
            f.write(f"{file}\n")
    with open(args.output_valid, "w") as f:
        for file in valids:
            f.write(f"{file}\n")
    with open(args.output_eval, "w") as f:
        for file in evals:
            f.write(f"{file}\n")


if __name__ == "__main__":
    main()
