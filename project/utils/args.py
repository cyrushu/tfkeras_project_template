import argparse


def Parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c","--config",dest="config", type=str,help="The Configuration file")
    args = parser.parse_args()
    return args
