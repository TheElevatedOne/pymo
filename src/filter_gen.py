import os
import cv2
import numpy
import spandrel
import torch.cuda as tc
import torch
from PIL import Image, ImageEnhance
from queue import Queue
from tqdm import tqdm
from .cuda.pycuda_denoise import PyCudaDenoise
from spandrel import ImageModelDescriptor, ModelLoader
from torchvision.transforms import transforms


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
            i = cv2.fastNlMeansDenoising(i, None, 10, 1, 12)
        else:
            pycuda = PyCudaDenoise(i, model)
            i = pycuda.run()

        cv2.imwrite(os.path.join(self.tdir, "denoise", img), i)

    def super_res(self, imgs: list, weight: str, pdir: str, threads: int) -> None:
        chunks = self.chunks(imgs, 1)
        manager = torch.multiprocessing.Manager()
        queue = manager.Queue()
        jobs = []
        torch.multiprocessing.set_start_method("spawn", force=True)

        for chunk in chunks:
            j = torch.multiprocessing.Process(target=self.sr_process, args=(chunk, queue, pdir, weight))
            jobs.append(j)
            j.start()

        pbar = tqdm(range(len(imgs)), desc="    03. Running Denoise (GPU-SR)")
        while pbar.n != len(imgs):
            if queue.get() == 0:
                pbar.update()
        pbar.close()

    def chunks(self, lst: list, n: int) -> list:
        h = len(lst) // n
        return [lst[i:i + h] for i in range(0, len(lst), h)]

    def sr_process(self, imgs: list, queue: Queue, pdir: str, weight: str) -> None:
        for img in imgs:
            model = ModelLoader().load_from_file(os.path.join(pdir, os.path.pardir, "weights", f"{weight}.pth"))
            model.cuda().eval()  # Prepare weights to use GPU
            assert isinstance(model, ImageModelDescriptor)
            img_arr = cv2.imread(os.path.join(self.tdir, "filter", img))
            img_tensor = transforms.Compose(
                [transforms.ToTensor()]
            )(img_arr).unsqueeze(0).cuda()

            with torch.no_grad():
                img_tensor = model(img_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
            img_res = numpy.uint8(img_tensor * 255)
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(self.tdir, "denoise", img), img_res)
            queue.put(0)