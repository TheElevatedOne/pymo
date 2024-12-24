<h1 align="center">PyMo (Python Motion Visualizer GUI)</h1>
<h4 align="center">A CLI script for visualizing differences between video frames, eg. motion.</h4>
<br>
<p align="center"><img src="logo.svg" width="40%"></p>

---

## Preview:

<details>
     <summary>test_01</summary>
     <video src="https://github.com/user-attachments/assets/fb343f2b-77e3-499a-925a-2cc816e2bcc5"></video>
     <hr>
     <h3>KNN PyCUDA</h3>
     <video src="https://github.com/user-attachments/assets/865683c9-b03c-43d7-a301-5b31e13161e5"></video>
     <h3>NLM2 PyCUDA</h3>
     <video src="https://github.com/user-attachments/assets/ff525d74-a4f6-42ed-8bda-7c2b3789256d"></video>
     <h3>CPU</h3>
     <video src="https://github.com/user-attachments/assets/39a1633f-248f-41e4-bab0-532e9125a24e"></video>
</details>

<details>
     <summary>test_02</summary>
     <video src="https://github.com/user-attachments/assets/104d2347-27bf-44f2-8418-94abc265527b"></video>
     <hr>
     <h3>KNN PyCUDA</h3>
     <video src="https://github.com/user-attachments/assets/a3b082ca-35d3-4b3d-8593-68edb36dd65e"></video>
     <h3>NLM2 PyCUDA</h3>
     <video src="https://github.com/user-attachments/assets/72ddcabe-0c50-4719-8c85-0aa36803ba5c"></video>
     <h3>CPU</h3>
     <video src="https://github.com/user-attachments/assets/deeb1c6f-94c2-46e4-8f69-f4a36ed08653"></video>
</details>

---

## Requirements:
- Python 3.10 or newer
- Nvidia GPU
  - Minimum: GTX 1650 4GB
  - Recommended: RTX 4060 8GB
- At least 16GB of RAM

---

## Installation:
```bash
git clone -b 1.1.0 https://github.com/TheElevatedOne/pymo.git && cd pymo
./install-venv.sh
```

---

## Running:
```
usage: pymo [-h] -i INPUT [-o PATH] [-n NAME] [-f INT[1, 50]] [-t CPU[1, 8]] [-s] [-c] [-m {knn,nlm,sr}] [-sr {Directory Empty!}]

     _____       __  __       
    |  __ \     |  \/  |      
    | |__) |   _| \  / | ___  
    |  ___/ | | | |\/| |/ _ \ 
    | |   | |_| | |  | | (_) |
    |_|    \__, |_|  |_|\___/ 
            __/ |             
           |___/              

   Python Motion Visualizer CLI

options:
  -h, --help            show this help message and exit

Required:
  Required Arguments

  -i INPUT, --input INPUT
                        Relative path to the Input Video

Optional:
  Optional Arguments

  -o PATH, --output PATH
                        (Optional) Absolute path to output directory
  -n NAME, --name NAME  (Optional) Custom Filename for the video
  -f INT[1, 50], --offset INT[1, 50]
                        (Optional) Number of Offset Frames [Default = 5]
  -t CPU[1, 8], --threads CPU[1, 8]
                        (Optional) Amount of threads to run the process on [Default = 2]
  -s, --slow_motion     (Optional) Sets the FPS of the Output Video to half the original;
                                   Essentially creating a slow-motion of the original without interpolation

Denoising:
  Denoising Arguments

  -c, --cpu             (Optional) Denoising step by default runs on CUDA Acceleration (if Nvidia GPU Available);
                                   Setting this makes it run on CPU even if GPU is Available
  -m {knn,nlm,sr}, --model {knn,nlm,sr}
                        (Optional) Model to use when denoising via GPU [Default = knn];
                                   SR - Super Resolution (ESRGAN, SwinIR, ...), 
                                   Needs a PyTorch weight file (.pth) in ./weights/ to be active.
  -sr {Directory Empty!}, --super_resolution {Directory Empty!}
                        (Optional) Choosing Weights for SR (if available)
                                   [Default = Directory Empty!]

```

### Detailed info on some arguments:
- `-o/--output [PATH]` An absolute path to a directory, e.g. on Linux `/home/${whoami}/directory`
  - If left empty, it will default to Input Video Dir
- `-n/--name [NAME]` Custom name for the output video, without extension. Default is `pymo_{input-vid-name}`
- `-f/--offset [OFFSET]` Amount of frames to offset the Visualizer
  - 1 - 3 -> Fast Moving object in the video
  - 4 - 6 -> "Normal" Moving object in the video
  - 7 - 8 -> Slow Moving object in the video
  - 9+ -> Really slow moving object in a video
  - Offset Frames are **REMOVED** from the video, due to the process. Amount of frames removed -> `OFFSET * 2`
- `-t/--threads` A few operations are single-threaded by default, so I multithreaded them
  - Try to keep it at the Maximum of the threads of your CPU, otherwise it may struggle
  - A Thread uses 2 - 4 GB of RAM, so keep in mind the amount of your RAM
- `-s/--slow_motion` Cuts the FPS in half to render it in Slow Motion
  - ex. 60 fps Video -> 30 fps Video but Twice the length
- `-c/--cpu` The program uses [PyCuda](https://pypi.org/project/pycuda/) for GPU Acceleration on Denoising by default (unless it does not detect a GPU)
  - This forces the Program to use the CPU, which results in higher quality render, but with a long render time.
- `-m/--model` GPU Denoising Model switcher.
  - KNN/NLM2 are good for some things, they are fast but sacrifice quality.
  - SR (Super Resolution) models (or weights) are slower than the above, but faster than CPU and have amazing quality.
    - They are trained weights on datasets (AI), which can do many things. I mainly use them to Denoise.
    - They need fast GPU's to run and a lot of VRAM. This may be the choice for someone with a good GPU.
    - Weights are not packaged with the program, Download them [here](./weights/README.md).

---

## Optimization:
As I develop this project, I try to optimize any algorithm I use after I got it working.<br>
With that, I just make 80% improvement on the Motion Generation (Difference Blend) and 50% improvement on CPU denoising.

Current Times (On My Machine):
- Specs:
  - CPU - Intel i5-9300H
  - GPU - Nvidia GTX 1650 Mobile
- Times (for Compute Intesive Tasks) [using test_01.mp4 | 4K, 210 Frames]:
  - Exporting Frames of the Video - 44s
  - Motion Generation - 1m 11s
  - Contrast Filter - 27s
  - Denoising:
    - CPU - 6m 2s
    - KNN - 44s
    - NLM2 - 54s
    - SR (1x-WB-Denoise.pth) - 6m 52s
  - Encoding Video - 17s

---

## Here's the video by which I found out about this: 
### [How To Make A Legit Sound Camera by @BennJordan](https://www.youtube.com/watch?v=c5ynZ3lMQJc) <br><br>

---

## Modules and scripts used:
- [**tqdm**](https://pypi.org/project/tqdm/) for progress bars
- [**PIL/Pillow**](https://pypi.org/project/pillow/) for easy image transformation
- [**opencv-python**](https://pypi.org/project/opencv-python/) for simpler video transformation, reading, writing and denoising
- [**imageio**](https://pypi.org/project/imageio/) for reading and writing images
- [**numpy**](https://pypi.org/project/numpy/) for converting Pillow images to cv2 arrays and vice versa
- [**pycuda**](https://pypi.org/project/pycuda/) for running GPU-accelerated Denoising
- [**torch**](https://pypi.org/project/torch/) for running GPU Super Resolution Models for Denoising
- [**torchvision**](https://pypi.org/project/torchvision/) for converting numpy arrays into tensors and vice versa
- [**spandrel**](https://pypi.org/project/spandrel/) project of chaiNNer developer, for reading all types of SR weights
- [**PyCuda_Denoise_Filters**](https://github.com/AlainPaillou/PyCuda_Denoise_Filters) for GPU denoising
  - Both of the `Mono` scripts were used in my project and were modified to fit the use-case
- [**Nuitka**](https://pypi.org/project/Nuitka/) for compiling `main.py` into a binary for easier running and adding to `$PATH`
<br>
<br>
