import argparse

import pandas as pd


def get_baseline():
    return pd.read_csv("baseline.csv", sep=";")

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--print-logs", type=str, required=False, choices=["no", "yes"], default="no")

    return parser.parse_args()
