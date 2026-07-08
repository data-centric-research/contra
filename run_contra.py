import os

from args_paser import parse_args
from core_model.core import execute


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    execute(args)


if __name__ == "__main__":
    main()
