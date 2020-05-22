import os
import argparse

from train import Train


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='preparation')
    parser.add_argument('data_path', action="store")
    parser.add_argument('model_pathname', action="store")
    parser.add_argument('out_path', action="store")
    parser.add_argument('--epochs', action="store", default=100 )

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    obj = Train( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()

