"""Pack the results to send back: results_gpu_<date>.zip (evaluation JSONs, model configs, logs; no weights)."""
import os, glob, time, zipfile
name = f"results_gpu_{time.strftime('%Y%m%d_%H%M')}.zip"
files = (glob.glob("results/eval_*.json") + glob.glob("results/numbers_gpu.json") + glob.glob("results/models_gpu/*.json")
         + glob.glob("logs/*.json") + glob.glob("logs/*.csv") + glob.glob("logs/*.txt") + glob.glob("logs/*.log"))
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(set(files)):
        z.write(f)
print("packed", name, len(files), "files", round(os.path.getsize(name) / 2 ** 20, 1), "MB")
