import cv2
import torch
import tqdm
import numpy as np
import HitnetModule
from pathlib import Path
import multiprocessing as mp

from dataset.utils import np2torch


def process(file_path):
    while True:
        for ids, lock in enumerate(process.lock_list):
            if lock.acquire(block=False):
                disp_path = (process.root / "disp_occ" / file_path).with_suffix(".png")
                dxy_path = (process.root / "slant_window" / file_path).with_suffix(
                    ".npy"
                )
                dxy_path.parent.mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    x = (
                        np2torch(
                            cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(
                                np.float32
                            )
                            / 256
                        )
                        .unsqueeze(0)
                        .cuda(ids)
                    )
                    x = HitnetModule.plane_fitting(x, 1024, 1, 9, 1e-3, 1e5)
                    x = x[0].cpu().numpy()
                np.save(dxy_path, x)
                lock.release()
                return


def process_init(lock_list, root):
    process.lock_list = lock_list
    process.root = root


def main(root, list_path):
    root = Path(root)

    with open(list_path, "rt") as fp:
        file_list = [Path(line.strip()) for line in fp]

    lock_list = [mp.Lock() for _ in range(8)]
    with mp.Pool(8, process_init, [lock_list, root]) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process, file_list), total=len(file_list)))


if __name__ == "__main__":
    main("/home/tiger/KITTIStereo/KITTI2012/training", "lists/kitti2012_train.list")
