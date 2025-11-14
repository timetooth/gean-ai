# app.py
# Streamlit UI for training + inference on CycleGAN_Simple/SE
import io
import os
import sys
import json
import time
import shutil
import zipfile
import subprocess
from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PY = BASE_DIR / "train.py"
TEST_PY  = BASE_DIR / "test.py"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR   = BASE_DIR / "output"        # train.py writes here
MODELS_DIR   = BASE_DIR / "models"        # we’ll archive trained checkpoints here
RESULTS_DIR  = BASE_DIR / "results"       # default folder for test.py runs

for d in (DATASETS_DIR, OUTPUT_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- small helpers ----------
def python_bin() -> str:
    return sys.executable  # whichever Python is running Streamlit

@contextmanager
def status_box(title: str):
    """Use st.status if available, else fallback to spinner."""
    try:
        with st.status(title, expanded=True) as s:
            yield s
    except Exception:
        with st.spinner(title):
            class Dummy:  # no-op update
                def update(self, *_, **__): pass
            yield Dummy()

def stream_process(cmd, cwd=BASE_DIR):
    """Run a subprocess and stream stdout to the UI."""
    placeholder = st.empty()
    log_lines = []
    st.write(f"```bash\n$ {' '.join(cmd)}\n```")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in iter(proc.stdout.readline, ""):
        log_lines.append(line.rstrip("\n"))
        # Keep the box reasonably sized
        tail = log_lines[-400:]
        placeholder.code("\n".join(tail))
    proc.wait()
    rc = proc.returncode
    if rc == 0:
        st.success("Done.")
    else:
        st.error(f"Process exited with code {rc}")
    return rc

def list_model_dirs():
    """
    Each model directory under MODELS_DIR should contain:
      - netG_A2B.pth
      - netG_B2A.pth
      - (optional) meta.json
    """
    items = []
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_dir(): 
            continue
        a2b = p / "netG_A2B.pth"
        b2a = p / "netG_B2A.pth"
        if a2b.exists() and b2a.exists():
            meta = {}
            if (p / "meta.json").exists():
                try:
                    meta = json.loads((p / "meta.json").read_text())
                except Exception:
                    meta = {}
            label = meta.get("label") or p.name
            items.append((label, p))
    return items

def archive_latest_run(run_name: str, extra_meta: dict):
    """Copy OUTPUT_DIR/{netG_*.pth} into models/run_name/, write meta."""
    target = MODELS_DIR / run_name
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in ("netG_A2B.pth", "netG_B2A.pth", "netD_A.pth", "netD_B.pth"):
        src = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, target / fname)
            copied.append(fname)
    meta = {"label": run_name, "time": int(time.time())}
    meta.update(extra_meta or {})
    (target / "meta.json").write_text(json.dumps(meta, indent=2))
    return target, copied

def save_uploaded_zip(file, name_hint: str):
    """Unzip a dataset into datasets/<name>/, return the path."""
    name = name_hint.strip().replace(" ", "_")
    if not name:
        name = f"dataset_{int(time.time())}"
    dest = DATASETS_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    # Write and unzip
    buf = io.BytesIO(file.getbuffer())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest)
    return dest

# ---------- UI ----------
st.set_page_config(page_title="CycleGAN Trainer/Inferencer", layout="wide")
st.title("CycleGAN — Train & Inference")

tabs = st.tabs(["🔎 Inference", "🛠️ Train new model", "ℹ️ Help"])

# ====================== Inference Tab ======================
with tabs[0]:
    st.subheader("Pick a model")
    models = list_model_dirs()
    cols = st.columns([3,1])
    with cols[0]:
        if not models:
            st.warning("No archived models in `models/` yet. Train one in the next tab, "
                       "or manually copy your checkpoints to `models/<name>/netG_A2B.pth` & `netG_B2A.pth`.")
        else:
            labels = [m[0] for m in models]
            choice = st.selectbox("Available models", labels)
            chosen_dir = dict(models)[choice]
            st.caption(f"Directory: `{chosen_dir}`")
    with cols[1]:
        if st.button("↻ Refresh list"):
            st.experimental_rerun()

    st.divider()
    st.subheader("Mode")

    mode = st.radio("Choose inference mode", ["Single image (A→B or B→A)", "Whole test set using test.py"], index=0)

    if mode == "Single image (A→B or B→A)":
        st.write("Upload one image and choose direction. The app will load the matching generator and display the result.")
        direction = st.selectbox("Direction", ["A2B (e.g., noisy→clean)", "B2A (e.g., clean→noisy)"])
        size = st.number_input("Resize to (px)", min_value=64, max_value=1024, value=256, step=64)
        up = st.file_uploader("Image", type=["png","jpg","jpeg","bmp","tiff"])
        run_btn = st.button("Run inference")

        if run_btn:
            if not models:
                st.error("No model found.")
            elif not up:
                st.error("Please upload an image.")
            else:
                # Lazy-import torch/torchvision to let the app load fast
                import torch
                import torchvision.transforms as T

                # Figure out which checkpoint to use
                ckpt = (chosen_dir / "netG_A2B.pth") if direction.startswith("A2B") else (chosen_dir / "netG_B2A.pth")

                # Import Generator as in repo
                sys.path.insert(0, str(BASE_DIR))
                from models import Generator  # signature (input_nc, output_nc)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                G = Generator(3,3).to(device).eval()
                try:
                    sd = torch.load(ckpt, map_location=device, weights_only=True)  # torch>=2.5
                except TypeError:
                    sd = torch.load(ckpt, map_location=device)
                G.load_state_dict(sd, strict=False)

                tfm = T.Compose([
                    T.Resize((size, size), interpolation=Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
                inv = T.Compose([
                    T.Lambda(lambda t: (t*0.5+0.5).clamp(0,1)),
                    T.ToPILImage()
                ])

                with status_box("Running inference…"):
                    img = Image.open(up).convert("RGB")
                    x = tfm(img).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        y = G(x).cpu().squeeze(0)
                    out = inv(y)
                    st.image([img, out], caption=["Input", f"Output ({ckpt.name})"], use_column_width=True)

                    # offer download
                    out_bytes = io.BytesIO()
                    out.save(out_bytes, format="PNG")
                    st.download_button("Download output PNG", out_bytes.getvalue(), file_name="output.png")

    else:
        st.write("Batch-translate a dataset using your `test.py` (expects `testA/` & `testB/`).")
        dataroot = st.text_input("Dataset root (folder that contains testA/ and testB/)", value=str(DATASETS_DIR / "noise2denoise"))
        which = st.selectbox("Direction", ["A2B","B2A","both"], index=2)
        results_dir = st.text_input("Results output folder", value=str(RESULTS_DIR / "batch_run"))
        n_cpu = st.number_input("DataLoader workers (set 0 if issues)", min_value=0, max_value=16, value=8, step=1)
        go = st.button("Run batch inference")

        if go:
            if not models:
                st.error("No model found.")
            else:
                args = [
                    python_bin(), str(TEST_PY),
                    "--dataroot", dataroot,
                    "--which_direction", which,
                    "--results_dir", results_dir,
                    "--n_cpu", str(n_cpu),
                    "--generator_A2B", str(chosen_dir / "netG_A2B.pth"),
                    "--generator_B2A", str(chosen_dir / "netG_B2A.pth"),
                ]
                # use GPU if available
                if shutil.which("nvidia-smi"):
                    args.append("--cuda")

                with status_box("Batch inference running…"):
                    stream_process(args)
                st.info(f"Images saved under: `{results_dir}`")

# ====================== Train Tab ======================
with tabs[1]:
    st.subheader("Dataset")
    st.write("Provide a dataset path that has `trainA/ trainB/ testA/ testB/`, or upload a ZIP in that structure.")

    ds_cols = st.columns(2)
    with ds_cols[0]:
        dataset_path = st.text_input("Existing dataset path", value=str(DATASETS_DIR / "noise2denoise"))
    with ds_cols[1]:
        up_zip = st.file_uploader("…or upload ZIP", type=["zip"])
        zip_name = st.text_input("Name for uploaded dataset folder", value="custom_dataset")
        if up_zip and st.button("Unzip to datasets/"):
            with status_box("Unzipping dataset…"):
                dest = save_uploaded_zip(up_zip, zip_name)
                st.success(f"Unzipped to: `{dest}`")
                dataset_path = str(dest)

    st.divider()
    st.subheader("Training parameters")

    c1, c2, c3 = st.columns(3)
    with c1:
        run_name = st.text_input("Run name (for archiving checkpoints)", value=time.strftime("run_%Y%m%d_%H%M%S"))
        batch_size = st.number_input("Batch size", 1, 16, 1)
        size = st.number_input("Image size", 64, 1024, 256, step=64)
    with c2:
        n_epochs = st.number_input("Total epochs", 1, 1000, 200)
        decay_epoch = st.number_input("Start LR decay at epoch", 1, 999, 100)
        lr = st.number_input("Learning rate", 1e-6, 1.0, 0.0002, format="%.6f")
    with c3:
        input_nc = st.number_input("Input channels", 1, 3, 3)
        output_nc = st.number_input("Output channels", 1, 3, 3)
        use_cuda = st.checkbox("Use CUDA", value=True)

    start_btn = st.button("Start training")

    if start_btn:
        # Ensure output dir exists; train.py expects it at save-time
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args = [
            python_bin(), str(TRAIN_PY),
            "--dataroot", dataset_path,
            "--batchSize", str(batch_size),
            "--n_epochs", str(n_epochs),
            "--decay_epoch", str(decay_epoch),
            "--size", str(size),
            "--input_nc", str(input_nc),
            "--output_nc", str(output_nc),
            "--lr", str(lr),
        ]
        if use_cuda and shutil.which("nvidia-smi"):
            args.append("--cuda")

        with status_box("Training… (logs stream below)"):
            rc = stream_process(args)

        if rc == 0:
            with status_box("Archiving checkpoints into models/…"):
                meta = {
                    "dataset": dataset_path,
                    "batchSize": batch_size,
                    "n_epochs": n_epochs,
                    "decay_epoch": decay_epoch,
                    "size": size,
                    "input_nc": input_nc,
                    "output_nc": output_nc,
                    "lr": lr,
                }
                target_dir, files = archive_latest_run(run_name, meta)
                if files:
                    st.success(f"Saved {files} to `{target_dir}`")
                    st.info("Use the Inference tab → Refresh to see the new model.")
                else:
                    st.warning("No checkpoint files found in output/. Did training finish and save?")

# ====================== Help Tab ======================
with tabs[2]:
    st.markdown("""**Dataset structure expected**""")


