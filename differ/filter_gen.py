import os
import cv2
from PIL import Image, ImageEnhance
from queue import Queue
from .cuda.pycuda_denoise import PyCudaDenoise
import torch.cuda as tc


class FilterGen:
    def __init__(self, temp_dir: str, cpu: bool) -> None:
        self.tdir = temp_dir
        self.cpu = cpu

    def contrast(self, imgs: list, queue: Queue) -> None:
        for img in imgs:
            i = Image.open(os.path.join(self.tdir, "diff", img)).convert("L")
            i = ImageEnhance.Contrast(i).enhance(5)
            i.save(os.path.join(self.tdir, "filter", img))
            queue.put(0)

    def denoise(self, img: str, model: str) -> None:
        i = cv2.imread(os.path.join(self.tdir, "filter", img), cv2.IMREAD_GRAYSCALE)

        if self.cpu or not tc.is_available():
            i = cv2.fastNlMeansDenoising(i, None, 10, 7, 21)
        else:
            pycuda = PyCudaDenoise(i, model)
            i = pycuda.run()

        cv2.imwrite(os.path.join(self.tdir, "denoise", img), i)