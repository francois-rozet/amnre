#!/usr/bin/env python

import pandas as pd
import glob

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Merge CSV files')
    parser.add_argument('pattern', help='input files (CSV) pattern')
    parser.add_argument('output', help='output file name')
    args = parser.parse_args()

    dfs = [
        pd.read_csv(filename)
        for filename in glob.glob(args.pattern)
    ]

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(args.output, index=False)
