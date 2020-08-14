import sys, os
from pathlib import Path

TRAIN_PATH = str(Path(__file__).absolute().parent.parent.parent.parent)
sys.path.insert(0, TRAIN_PATH)
