#!/usr/bin/env python3
"""
Rename sub-directories: replace every occurrence of 'RandX' in a
directory name with 'ZeroX'.

Usage (for first level sub-directories):
    python rename_randx_to_zerox.py /path/to/parent_dir

For all sub-directories:
    python rename_randx_to_zerox.py /path/to/parent_dir --recursive
"""

from pathlib import Path
import argparse
import sys

def rename_subdirs(parent: Path, recursive: bool = False, before='RandX', after="ZeroX") -> None:
    f"""Rename sub-directories whose names contain {before}."""
    if not parent.is_dir():
        print(f"Error: {parent} is not a directory", file=sys.stderr)
        sys.exit(1)

    itr = parent.rglob("*") if recursive else parent.iterdir()

    for p in itr:
        if not p.is_dir():                 # 只重命名目录
            continue
        if before not in p.name:
            continue

        new_name = p.name.replace(before, after)
        target = p.with_name(new_name)

        if target.exists():
            print(f"Skip: {target} already exists; {p} unchanged")
            continue

        p.rename(target)
        print(f"{p.relative_to(parent)}  ->  {target.relative_to(parent)}")

if __name__ == "__main__":

    rename_subdirs(Path('./results/training/Math-L2O-P/').expanduser().resolve(), False)
