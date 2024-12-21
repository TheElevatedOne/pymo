import os.path as op
from tqdm import tqdm
from PIL import Image


class FrameGen:
    def __init__(self, width: int, height: int, offset: int, tdir: str, length: int, spacer: str, part: int) -> None:
        self.width = width
        self.height = height
        self.offset = offset
        self.tdir = tdir
        self.length = length
        self.spacer = spacer
        self.part = part

        self.generate()

    def generate(self) -> None:
        im = Image.new(mode="RGB", size=(self.width, self.height), color=(0, 0, 0))
        for i in tqdm(range(self.offset), desc="        Generating Offset Frames %d/2" % self.part):
            im.save(op.join(self.tdir, self.spacer % self.length))
            self.length += 1
