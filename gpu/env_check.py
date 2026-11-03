"""Print and record the server environment (logs/env.json); exit 2 if a required package is missing."""
import os, sys, json, shutil, subprocess, platform
info = dict(python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(), cpus=os.cpu_count())
try:
    info["mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
except (ValueError, OSError):
    pass
info["disk_free_gb"] = round(shutil.disk_usage(".").free / 2 ** 30, 1)
missing = []
for mod in ["numpy", "scipy", "sklearn", "skimage", "torch"]:
    try:
        m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
    except Exception as e:
        info[mod] = None; missing.append(mod)
try:
    import torch
    info["cuda"] = torch.cuda.is_available()
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["torch_cuda"] = torch.version.cuda
except Exception:
    pass
try:
    info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                         "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
except Exception as e:
    info["nvidia_smi"] = repr(e)
info["gpu_hold"] = os.path.exists(os.path.expanduser("~/gpu_hold.py"))
os.makedirs("logs", exist_ok=True)
json.dump(info, open("logs/env.json", "w"), indent=1)
print(json.dumps(info, indent=1))
if missing:
    print("MISSING PACKAGES:", missing)
    print("install with:  python3 -m pip install --user " + " ".join({"sklearn": "scikit-learn", "skimage": "scikit-image"}.get(m, m) for m in missing))
    sys.exit(2)
if not info.get("cuda"):
    print("WARNING: torch sees no CUDA device")
    sys.exit(3)
