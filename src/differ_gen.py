import os
import numpy as np
from queue import Queue
import imageio.v3 as imio3

class DifferGen:
    def __init__(self, temp_dir: str) -> None:
        """
        class DifferGen requires Temporary Directory path as an Initialization Variable
        """
        self.tdir = temp_dir

    def run(self, imgs: list, queue: Queue) -> None:
        for img in imgs:
            bt, tp = os.path.join(self.tdir, "bottom", img), os.path.join(self.tdir, "top", img)
            # bottom_raw = Image.open(bt).convert("RGBA")
            # top_raw = Image.open(tp).convert("RGBA")
            #
            # bottom_npy = numpy.array(bottom_raw)
            # top_npy = numpy.array(top_raw)
            #
            # bottom_fl = bottom_npy.astype(float)
            # top_fl = top_npy.astype(float)
            #
            # diff_fl = difference(bottom_fl, top_fl, 1)
            # diff_npy = numpy.uint8(diff_fl)
            # diff_raw = Image.fromarray(diff_npy)
            #
            # diff_raw.save(os.path.join(self.tdir, "diff", img))

            bottom, top = np.int16(imio3.imread(bt)), np.int16(imio3.imread(tp))
            diff = np.uint8(np.abs(bottom - top))
            imio3.imwrite(os.path.join(self.tdir, "diff", img), diff)

            queue.put(0)