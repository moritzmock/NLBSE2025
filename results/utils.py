import argparse

import pandas as pd


def get_baseline():
    return pd.read_csv("baseline.csv", sep=";")

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, required=True)

    return parser.parse_args()
