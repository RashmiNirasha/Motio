"""
One-command launcher for the Motio live-monitoring MVP.

    python run.py              set up, (train if needed), start server, open browser
    python run.py --retrain    force retraining the binary model
    python run.py --tune       re-tune the decision threshold after training
    python run.py --port 8080  serve on a different port
    python run.py --no-browser don't open the browser automatically

What it does, in order:
  1. Creates a local virtual environment in .venv (first run only)
  2. Installs the MVP dependencies into it (re-run only when this list changes)
  3. Makes sure mvp/models_binary/child_object_interaction.pt exists,
     building the temporal dataset and training the model if necessary
  4. Starts the FastAPI server and opens the live model page
"""

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
import venv
import webbrowser


ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, ".venv")
STAMP_PATH = os.path.join(VENV_DIR, ".motio-deps")

MODEL_DIR = os.path.join(ROOT, "mvp", "models_binary")
MODEL_PATH = os.path.join(MODEL_DIR, "child_object_interaction.pt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "iphone_threshold.txt")
DATASET_DIR = os.path.join(ROOT, "mvp", "dataset_v2")
RAW_DATA_DIR = os.path.join(ROOT, "data", "childplay_hand", "data")

# Everything needed to build the dataset, train, tune and serve the MVP.
DEPENDENCIES = [
    "fastapi",
    "uvicorn",
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "requests",
]

LIVE_PAGE = "/app/live-model.html"


def log(message):
    print(f"\n[motio] {message}", flush=True)


def fail(message):
    print(f"\n[motio] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def in_venv():
    return os.path.abspath(sys.prefix) == os.path.abspath(VENV_DIR)


# --------------------------------------------------
# 1 + 2. Environment
# --------------------------------------------------

def ensure_venv():
    """Create .venv if needed and re-launch this script inside it."""
    if in_venv():
        return

    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ is required (found {sys.version.split()[0]}).")

    if not os.path.exists(venv_python()):
        log("Creating virtual environment in .venv ...")
        venv.create(VENV_DIR, with_pip=True)

    result = subprocess.run([venv_python(), __file__, *sys.argv[1:]], cwd=ROOT)
    sys.exit(result.returncode)


def ensure_dependencies():
    wanted = hashlib.sha1("\n".join(DEPENDENCIES).encode()).hexdigest()

    if os.path.exists(STAMP_PATH):
        with open(STAMP_PATH) as f:
            if f.read().strip() == wanted:
                return

    log("Installing dependencies (first run can take a few minutes) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *DEPENDENCIES],
        check=True,
    )

    with open(STAMP_PATH, "w") as f:
        f.write(wanted)


# --------------------------------------------------
# 3. Model
# --------------------------------------------------

def is_lfs_pointer(path):
    """True if the file is a Git LFS placeholder rather than real data."""
    try:
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def dataset_ready():
    for split in ("train", "val", "test"):
        for prefix in ("X", "y"):
            path = os.path.join(DATASET_DIR, f"{prefix}_{split}.npy")
            if not os.path.exists(path) or is_lfs_pointer(path):
                return False
    return True


def run_step(description, script):
    log(description)
    subprocess.run(
        [sys.executable, os.path.join("mvp", script)],
        cwd=ROOT,
        check=True,
    )


def ensure_model(retrain, tune):
    model_ok = os.path.exists(MODEL_PATH) and not is_lfs_pointer(MODEL_PATH)

    if model_ok and not retrain:
        log("Trained model found.")
    else:
        if not dataset_ready():
            if os.path.isdir(RAW_DATA_DIR):
                run_step(
                    "Building temporal dataset ...",
                    "build_temporal_dataset_v2.py",
                )
            else:
                fail(
                    "No trained model and no dataset to train one from.\n"
                    "  Do ONE of the following, then run this again:\n"
                    "  - Copy child_object_interaction.pt into mvp/models_binary/\n"
                    "  - Run `git lfs install && git lfs pull` to fetch mvp/dataset_v2/\n"
                    "  - Put the ChildPlay-Hand data in data/childplay_hand/data/\n"
                    "    (see data/README.md)"
                )

        run_step("Training the binary interaction model ...", "train_binary_mvp.py")

    # The chosen threshold (0.66) is committed, so only re-tune when asked.
    if tune or not os.path.exists(THRESHOLD_PATH):
        if not dataset_ready():
            fail("Threshold tuning needs mvp/dataset_v2/ (run `git lfs pull`).")
        run_step("Tuning the decision threshold ...", "tune_binary_threshold.py")


# --------------------------------------------------
# 4. Server
# --------------------------------------------------

def open_browser_when_ready(port):
    import urllib.request

    url = f"http://127.0.0.1:{port}"

    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        return

    log(f"Opening {url}{LIVE_PAGE}")
    webbrowser.open(url + LIVE_PAGE)


def start_server(port, open_browser):
    log(f"Starting server on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print(f"        Live model:  http://127.0.0.1:{port}{LIVE_PAGE}")
    print(f"        Camera test: http://127.0.0.1:{port}/app/camera-test.html")
    print(f"        Health:      http://127.0.0.1:{port}/health", flush=True)

    if open_browser:
        threading.Thread(
            target=open_browser_when_ready, args=(port,), daemon=True
        ).start()

    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn", "mvp.webapp.server:app",
                "--host", "0.0.0.0",
                "--port", str(port),
            ],
            cwd=ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        log("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run the Motio MVP.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--retrain", action="store_true", help="force retraining")
    parser.add_argument("--tune", action="store_true", help="re-tune the threshold")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ensure_venv()
    ensure_dependencies()

    try:
        ensure_model(args.retrain, args.tune)
    except subprocess.CalledProcessError as error:
        fail(f"Step failed with exit code {error.returncode}.")

    start_server(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
