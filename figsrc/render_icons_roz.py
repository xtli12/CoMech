"""Superior views of the reference foot with the five Rozansky morphologies of the atlas analogue (Fig. 5d icons;
Claude, 2026-11-03).  Same camera and colours as render_icons.py (pure geometry, no fields)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
from render_icons import superior, OUT
from scene_engine import Foot, Action
from exp.scenarios import COAL_MORPH

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for t in ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V"]:
        superior(Foot(**COAL_MORPH[t]), Action(), "icon_coal_" + t.replace(" ", ""))
