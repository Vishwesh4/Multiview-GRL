# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is to run training multi graph view representation learning using graphs using the modules
# --------------------------------------------------------------------------------------------------------------------------
import sys

from pathlib import Path
import yaml
import argparse

import utils
import trainer

parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config location",required=True)
parser.add_argument("-s", help="seed",default=0,type=int)
args = parser.parse_args()

config_path = args.c
deep_trainer = utils.TrainEngine_multiview(config_pth=config_path, random_seed=args.s)
deep_trainer.run()
