"""Prepare the pinned SAM2 backbone and original Fold 1 adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "local_models"
CHECKOUT = WEIGHTS / "sam2_repo"
BASE = WEIGHTS / "sam2.1_hiera_base_plus.pt"
IDENTITY = json.loads((ROOT / "configs/identity.json").read_text())
BACKBONE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_verified(stream, destination, expected):
    if destination.is_file() and digest(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with temporary.open("wb") as out:
            shutil.copyfileobj(stream, out)
        if digest(temporary) != expected:
            raise ValueError(f"Checksum mismatch for {destination.name}.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_adapters(archive):
    if digest(archive) != IDENTITY["archive_sha256"]:
        raise ValueError("Select the completed BALoRA_Full_Run_Results(1).zip from the Fold 1 study.")
    with ZipFile(archive) as bundle:
        for method, expected in IDENTITY["weight_sha256"].items():
            member = f"{IDENTITY['archive_prefix']}/{method}/research/best.pt"
            with bundle.open(member) as stream:
                write_verified(stream, WEIGHTS / f"{method}.pt", expected)


def setup_sam2():
    if not (CHECKOUT / ".git").is_dir():
        subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/facebookresearch/sam2.git",
                        str(CHECKOUT)], check=True)
    subprocess.run(["git", "checkout", "--detach", IDENTITY["sam2_commit"]], cwd=CHECKOUT, check=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=CHECKOUT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Completed Fold 1 results ZIP")
    parser.add_argument("--backbone", type=Path, help="Optional already-downloaded official checkpoint")
    args = parser.parse_args()
    extract_adapters(args.results)
    setup_sam2()
    if args.backbone:
        with args.backbone.open("rb") as stream:
            write_verified(stream, BASE, IDENTITY["backbone_sha256"])
    elif not BASE.is_file() or digest(BASE) != IDENTITY["backbone_sha256"]:
        with urlopen(BACKBONE_URL, timeout=60) as stream:
            write_verified(stream, BASE, IDENTITY["backbone_sha256"])
    print("Verified model files are ready. Start the app with: python -m streamlit run app.py --server.address 127.0.0.1")
