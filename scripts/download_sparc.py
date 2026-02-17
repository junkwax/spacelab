#!/usr/bin/env python3
"""
Download and extract the SPARC Rotmod_LTG dataset.

Source: Lelli, McGaugh & Schombert (2016), AJ 152, 157
        http://astroweb.case.edu/SPARC/

If you use this data in publications, please cite:
  Lelli F., McGaugh S.S., Schombert J.M., 2016, AJ, 152, 157

Usage:
    python scripts/download_sparc.py [--output-dir data/sparc]
"""

import os
import sys
import zipfile
import argparse
import urllib.request

SPARC_URL = "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip"
TABLE1_URL = "http://astroweb.case.edu/SPARC/Table1.mrt"


def download_sparc(output_dir: str = "data/sparc"):
    """Download and extract SPARC rotation curve data."""
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "Rotmod_LTG.zip")

    # Download the zip file
    print(f"Downloading SPARC data from {SPARC_URL}...")
    try:
        urllib.request.urlretrieve(SPARC_URL, zip_path)
        print(f"  Downloaded to {zip_path}")
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        print("  You can manually download from: http://astroweb.case.edu/SPARC/")
        print(f"  Place Rotmod_LTG.zip in {output_dir}/ and rerun with --extract-only")
        return False

    # Extract
    print(f"Extracting to {output_dir}/...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Count extracted files
    dat_files = [f for f in os.listdir(output_dir) if f.endswith("_rotmod.dat")]
    print(f"  Extracted {len(dat_files)} galaxy rotation curve files")

    # Download Table1 (galaxy properties)
    table1_path = os.path.join(output_dir, "Table1.mrt")
    print(f"Downloading galaxy properties table...")
    try:
        urllib.request.urlretrieve(TABLE1_URL, table1_path)
        print(f"  Downloaded Table1.mrt")
    except Exception as e:
        print(f"  Warning: Could not download Table1.mrt: {e}")

    print("\nDone! SPARC data is ready in", output_dir)
    print("Each file has columns:")
    print("  Rad [kpc] | Vobs [km/s] | errV [km/s] | Vgas [km/s] | "
          "Vdisk [km/s] | Vbul [km/s] | SBdisk [L/pc²] | SBbul [L/pc²]")
    return True


def extract_only(output_dir: str = "data/sparc"):
    """Extract a previously downloaded zip file."""
    zip_path = os.path.join(output_dir, "Rotmod_LTG.zip")
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found. Download it first.")
        return False

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    dat_files = [f for f in os.listdir(output_dir) if f.endswith("_rotmod.dat")]
    print(f"  Extracted {len(dat_files)} galaxy files")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SPARC rotation curve data")
    parser.add_argument("--output-dir", default="data/sparc", help="Output directory")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract an existing zip file")
    args = parser.parse_args()

    if args.extract_only:
        success = extract_only(args.output_dir)
    else:
        success = download_sparc(args.output_dir)

    sys.exit(0 if success else 1)
