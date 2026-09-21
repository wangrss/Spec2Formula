"""Optional accelerator; writes only to this checkout's _bin directory."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error('C++17 compiler not found. Use --cxx clang++ or install a compiler; the Python backend also works.')
    folder = root/'_bin'
    folder.mkdir(exist_ok=True)
    target = folder/('enumerate.exe' if os.name == 'nt' else 'enumerate')
    subprocess.run([compiler,'-std=c++17','-O3',str(root/'native/enumerate.cpp'),'-o',str(target)], check=True)
    print(target)

if __name__ == '__main__':
    main()

