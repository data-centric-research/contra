import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen_dataset.gen_imagefolder_exp_data import main


if __name__ == "__main__":
    main()
