import multiprocessing
import os
import shutil
import sys
import cv2
import time
import torch.cuda as tc
import termcolor
from tqdm import tqdm
from .differ_gen import DifferGen
from .frame_gen import FrameGen
from .filter_gen import FilterGen


class Difference:
    """Class containing the basis of the motion extraction process

    Arguments:
    input_path (str)  | Relative or Absolute path to Input Video
    output_path (str) | Absolute path to output directory;
                      | if None, base path of Input Video
    name (str)        | Custom Output Name; Defaults to "pymo_{input_name}.mp4"
    offset (int)      | Number of frames the Input Video to offset by
    threads (int)     | Number of threads to run the program at
    slomo (bool)      | Enable "slow-motion" / halving the fps of the output
    cpu (bool)        | Force use CPU to Denoise even if GPU available
    model (str)       | GPU Model to use for Denoising
    sr (str)          | If "sr" set in GPU Model,
                      | Enables Super-Resolution AI Models;
                      | This argument specifies the model.
    """
    def __init__(self, input_path: str, output_path: str, name: str, offset: int, threads: int, slomo: bool, cpu: bool, model: str, sr: str) -> None:
        self.inp = os.path.abspath(input_path)
        self.oup = os.path.dirname(os.path.abspath(input_path))
        if output_path is not None: self.oup = os.path.abspath(output_path)

        if name is not None: self.name = f"{name}.mp4"
        self.name = f"pymo_{os.path.splitext(input_path)[0]}.mp4"

        self.offset = offset
        self.pydir = os.path.dirname(__file__)
        self.tdir = os.path.join(self.pydir, "temp")

        self.threads = threads
        self.slomo = slomo
        self.cpu = cpu
        self.model = model
        self.sr = sr

        # Exceptions, Warnings and General Stuff
        if not os.path.isfile(self.inp):
            print("----------------------------------")
            print(f'>>> {termcolor.colored("Input Path Not Valid - File Does Not Exist", "red", attrs=["bold"])}')
            print(f'>>> {termcolor.colored("Program will now exit.", attrs=["bold"])}')
            sys.exit(5)

        if self.model == "sr" and self.sr == "Directory Empty!":
            print(f'>>> {termcolor.colored("UserWarning:", "red", "on_black", ["bold"])} Argument "-m/--model" has been '
                  f'set to "sr" but no models found in "./weights/",')
            print('>>> Switching back to default model, e.g. "knn"')
            self.model = "knn"

    def temp_dir(self) -> None:
        """Creates a temporary directory; if exists, removes it"""
        if not os.path.isdir(self.tdir):
            os.mkdir(self.tdir)
        else:
            shutil.rmtree(self.tdir)

    def vid_frames(self) -> None:
        """Extracts frames of the Input Video to a predetermined directory"""
        video = cv2.VideoCapture(self.inp)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        wv, wh = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        spacer = f"%0{len(str(length))}d.png"

        # bottom
        print("    01. Preparing Frames")
        os.mkdir(os.path.join(self.tdir, "bottom"))
        os.mkdir(os.path.join(self.tdir, "top"))

        for i in tqdm(range(length), desc="        Exporting Frames"):
            s, img = video.read()
            cv2.imwrite(os.path.join(self.tdir, "bottom", spacer % i), img)

        frames = os.listdir(os.path.join(self.tdir, "bottom"))
        frames.sort()

        for i in tqdm(frames, desc="        Copying Frames"):
            shutil.copyfile(os.path.join(self.tdir, "bottom", i), os.path.join(self.tdir, "top", i))

        for i in tqdm(range(length - 1, -1, -1), desc="        Creating Space For Offset Frames"):
            os.rename(os.path.join(self.tdir, "top", spacer % i), os.path.join(self.tdir, "top", spacer % (i + self.offset)))

        FrameGen(wv, wh, self.offset, os.path.join(self.tdir, "bottom"), length, spacer, 1)

        FrameGen(wv, wh, self.offset, os.path.join(self.tdir, "top"), 0, spacer, 2)

        video.release()

    def dif_frames(self) -> None:
        """Applies Difference Blend, Contrast and Denoise to the Frames"""
        os.mkdir(os.path.join(self.tdir, "diff"))
        os.mkdir(os.path.join(self.tdir, "filter"))
        os.mkdir(os.path.join(self.tdir, "denoise"))
        bottom, top = os.listdir(os.path.join(self.tdir, "bottom")), os.listdir(os.path.join(self.tdir, "top"))
        bottom.sort()

        # Running Threaded Difference Blend
        diff_gen = DifferGen(self.tdir)
        diff_ch = self.chunks(bottom, self.threads)
        diff_jobs = []

        manager = multiprocessing.Manager()
        queue = manager.Queue()

        for ch in diff_ch:
            j = multiprocessing.Process(target=diff_gen.run, args=(ch, queue))
            diff_jobs.append(j)
            j.start()

        diff_prog = tqdm(range(len(bottom)), desc="    01. Generating Motion Frames")

        while diff_prog.n != len(bottom):
            if queue.get() == 0:
                diff_prog.update()

        diff_prog.close()

        # diff_prog.disable = True

        # Running Contrast Filter
        filter_gen = FilterGen(self.tdir, self.cpu)
        filter_ch = self.chunks(bottom, self.threads)
        filter_jobs = []

        queue = manager.Queue()

        for ch in filter_ch:
            j = multiprocessing.Process(target=filter_gen.contrast, args=(ch, queue))
            filter_jobs.append(j)
            j.start()

        filter_prog = tqdm(range(len(bottom)), desc="    02. Running Contrast Filter")

        while filter_prog.n != len(bottom):
            if queue.get() == 0:
                filter_prog.update()

        filter_prog.close()

        # filter_prog.disable = True

        # Running Denoise
        if self.model != "sr":
            compute = "GPU" if not self.cpu and tc.is_available() else "CPU"
            for img in tqdm(bottom, desc="    03. Running Denoise (%s)" % compute):
                filter_gen.denoise(img, self.model)
        else:
            filter_gen.super_res(bottom, self.sr, self.pydir, self.threads)

    def gen_video(self) -> str:
        """Encodes Video from Denoised Frames"""
        frames = os.listdir(os.path.join(self.tdir, "denoise"))
        frames.sort()
        frames = frames[self.offset: -self.offset]

        orig = cv2.VideoCapture(self.inp)
        wv, wh = int(orig.get(cv2.CAP_PROP_FRAME_WIDTH)), int(orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = orig.get(cv2.CAP_PROP_FPS)
        if self.slomo:
            fps /= 2

        orig.release()

        result = cv2.VideoWriter(os.path.join(self.oup, self.name), cv2.VideoWriter.fourcc(*"mp4v"), fps, (wv, wh))

        for i in tqdm(frames, desc="    01. Encoding Video"):
            frame = cv2.imread(os.path.join(self.tdir, "denoise", i))
            result.write(frame)

        result.release()

        return os.path.join(self.oup, self.name)

    def chunks(self, lst: list, n: int) -> list:
        """List splitter into chunks for multithreaded workloads"""
        h = len(lst) // n
        return [lst[i:i + h] for i in range(0, len(lst), h)]
