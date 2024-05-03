import torch
import tqdm
import numpy as np
import HitnetModule
from pathlib import Path
import multiprocessing as mp
from PIL import Image

from dataset.utils import readPFM, np2torch


def process(file_path):
    
    lock = process.lock_list[0]
    if lock.acquire(block=True):
        base_path = process.root / file_path.parents[1]  # Get the base path (train or test)
        pfm_path = base_path / "disparity" / (file_path.stem + ".tiff")  # Updated extension and directory
        dxy_path = base_path / "slant" / (file_path.stem + ".npy")
        
        dxy_path.parent.mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            x = np2torch(np.array(Image.open(str(pfm_path)))).unsqueeze(0).cuda(0)
            x = HitnetModule.plane_fitting(x, 256, 0.1, 9, 0, 1e5)
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

    lock_list = [mp.Lock()]
    with mp.Pool(1, process_init, [lock_list, root]) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process, file_list), total=len(file_list)))


if __name__ == "__main__":
    main("stereo_dataset_256_128", "lists/train_new.txt")
    main("stereo_dataset_256_128", "lists/test_new.txt")
