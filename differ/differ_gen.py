import os
import numpy
from PIL import Image
from queue import Queue
from blend_modes import difference

class DifferGen:
    def __init__(self, temp_dir: str) -> None:
        """
        class DifferGen requires Temporary Directory path as an Initialization Variable
        """
        self.tdir = temp_dir

    def run(self, imgs: list, queue: Queue) -> None:
        for img in imgs:
            bt, tp = os.path.join(self.tdir, "bottom", img), os.path.join(self.tdir, "top", img)
            bottom_raw = Image.open(bt).convert("RGBA")
            top_raw = Image.open(tp).convert("RGBA")

            bottom_npy = numpy.array(bottom_raw)
            top_npy = numpy.array(top_raw)

            bottom_fl = bottom_npy.astype(float)
            top_fl = top_npy.astype(float)

            diff_fl = difference(bottom_fl, top_fl, 1)
            diff_npy = numpy.uint8(diff_fl)
            diff_raw = Image.fromarray(diff_npy)

            diff_raw.save(os.path.join(self.tdir, "diff", img))
            queue.put(0)