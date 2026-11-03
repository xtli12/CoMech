"""Supplementary Fig. 9: example CT readings (preliminary, AI-assisted) of four feet, P01-P30 identifiers only.
The 20 tiles in results/ct_tiles_s9/ are the de-identified PACS-export crops of the previous version of this figure,
extracted unchanged; this script only lays them out with the house style (no text over the images)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figstyle import *
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "supp_figures")
TILES = os.path.join(ROOT, "results", "ct_tiles_s9")
ROWS = [
    ("a", "P09-L, native coronal series", "type I (middle facet), medium confidence",
     ["Slice 388", "Slice 394", "Slice 399", "Slice 401", "Slice 403"],
     "Oblique, overlapping talus–sustentaculum interface with sclerotic irregular margins; posteromedial extension (left)."),
    ("b", "P05-R, pseudo-coronal reformat", "type I with posteromedial extension, medium confidence",
     ["s = 0.36", "s = 0.43", "s = 0.51", "s = 0.60", "s = 0.64"],
     "Straight, narrowed middle-facet interface (right two) and posteromedial ossicles."),
    ("c", "P13-R, pseudo-coronal reformat", "type III (posteromedial), medium confidence",
     ["s = 0.57", "s = 0.62", "s = 0.67", "s = 0.72", "s = 0.84"],
     "Ossicle wedged at the posteromedial end of the posterior facet; middle facet open (right)."),
    ("d", "P07-L, pseudo-coronal reformat", "indeterminate, low confidence",
     ["s = 0.68", "s = 0.84", "s = 1.06", "s = 1.28", "s = 1.39"],
     "Posterior and middle facets visible; coalition not confirmed on this export."),
]
TILE = 31.2                                  # mm
GAP = (FIG_W_SUPP - 5 * TILE) / 4
RH = 43.0                                    # row height (mm)
cv = Canvas(FIG_W_SUPP, 4 * RH - 1.0)
for r, (letter, title, reading, labels, note) in enumerate(ROWS):
    y = r * RH
    cv.letter(0, y + 0.5, letter)
    t = cv.header(4.8, y + 3.4, title + ":")
    cv.fig.canvas.draw()
    bb = t.get_window_extent(); w_mm = bb.width / cv.fig.dpi * 25.4
    cv.text(4.8 + w_mm + 1.2, y + 3.4, "preliminary reading, " + reading, fontsize=FS_LABEL, color=INK2)
    for c in range(5):
        x = c * (TILE + GAP)
        ax = cv.ax(x, y + 7.4, TILE, TILE)
        ax.imshow(Image.open(os.path.join(TILES, f"{letter}{c + 1}.png")), cmap="gray", interpolation="none")
        ax.set_axis_off()
        cv.text(x, y + 6.6, labels[c], fontsize=FS_SMALL, color=INK2)
    cv.text(0, y + 7.4 + TILE + 2.9, "← Medial", fontsize=FS_SMALL, color=INK2)
    cv.text(TILE + GAP, y + 7.4 + TILE + 2.9, note, fontsize=FS_SMALL, color=INK)
os.makedirs(OUT, exist_ok=True)
cv.save(os.path.join(OUT, "supp_fig9"))
print("saved supp_fig9")
