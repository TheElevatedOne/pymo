     _____       __  __       
    |  __ \     |  \/  |      
    | |__) |   _| \  / | ___  
    |  ___/ | | | |\/| |/ _ \ 
    | |   | |_| | |  | | (_) |
    |_|    \__, |_|  |_|\___/ 
            __/ |             
           |___/
# PyMo (Python Motion Visualizer GUI)
### A CLI script for visualizing differences between video frames, eg. motion.

---

## Preview

---

## Requirements:
- Python 3.10 or newer
- Nvidia GPU with installed drivers (Optional but Recommended)
- At least 16GB of RAM

---

## Installation:
```bash
git clone https://github.com/TheElevatedOne/pymo.git && cd pymo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running:
```
usage: main.py [-h] -i INPUT [-o OUTPUT] [-n NAME] [-f OFFSET] [-t THREADS] [-s] [-c] [-m MODEL]

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
  -i INPUT, --input INPUT
                        Relative path to the Input Video
  -o OUTPUT, --output OUTPUT
                        (Optional) Absolute path to output directory
  -n NAME, --name NAME  (Optional) Custom Filename for the video
  -f OFFSET, --offset OFFSET
                        (Optional) Number of Offset Frames [Default = 5]
  -t THREADS, --threads THREADS
                        (Optional) Amount of threads to run the process on [Default = 2]
  -s, --slow_motion     (Optional) Sets the FPS of the Output Video to half the original;
                        Essentially creating a slow-motion of the original without interpolation
  -c, --cpu             (Optional) Denoising step by default runs on CUDA Acceleration (if Nvidia GPU Available)
                        Setting this makes it run on CPU even if GPU is Available
  -m MODEL, --model MODEL
                        (Optional) Model to use when denoising via GPU [Default = nlm] [Options: nlm, knn]
```

### Detailed info on arguments:
- `-i/--input [FILE]` A Video File with OpenCV2 supported extensions
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
- `-m/--model` GPU Denoising Model switcher, both are good for some things
