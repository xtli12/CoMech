import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu import config as CFG
print(" ".join(CFG.RUNS))
