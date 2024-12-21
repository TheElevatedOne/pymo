import argparse
import sys
from differ.differ import Difference


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
     _____       __  __       
    |  __ \     |  \/  |      
    | |__) |   _| \  / | ___  
    |  ___/ | | | |\/| |/ _ \ 
    | |   | |_| | |  | | (_) |
    |_|    \__, |_|  |_|\___/ 
            __/ |             
           |___/              
           
   Python Motion Visualizer CLI""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--input", required=True, help="""Relative path to the Input Video""", type=str)
    parser.add_argument("-o", "--output", required=False, help="(Optional) Absolute path to output directory", type=str)
    parser.add_argument("-n", "--name", required=False, help="(Optional) Custom Filename for the video", type=str)
    parser.add_argument("-f", "--offset", required=False, default=5, help="(Optional) Number of Offset Frames [Default = 5]", type=int)
    parser.add_argument("-t", "--threads", required=False, default=2, help="(Optional) Amount of threads to run the process on [Default = 2]", type=int)
    parser.add_argument("-s", "--slow_motion", required=False, default=False, help="""(Optional) Sets the FPS of the Output Video to half the original;
Essentially creating a slow-motion of the original without interpolation""", action="store_true")
    parser.add_argument("-c", "--cpu", required=False, default=False, help="""(Optional) Denoising step by default runs on CUDA Acceleration (if Nvidia GPU Available);
Setting this makes it run on CPU even if GPU is Available""", action="store_true")
    parser.add_argument("-m", "--model", required=False, default="knn", help="(Optional) Model to use when denoising via GPU [Default = knn] [Options: nlm, knn]")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main() -> None:
    args = parse() # Parse CLI Arguments
    print("""
     _____       __  __       
    |  __ \     |  \/  |      
    | |__) |   _| \  / | ___  
    |  ___/ | | | |\/| |/ _ \ 
    | |   | |_| | |  | | (_) |
    |_|    \__, |_|  |_|\___/ 
            __/ |             
           |___/              
----------------------------------""")
    print("Initializing")
    di = Difference(args.input, args.output, args.name, args.offset, args.threads, args.slow_motion, args.cpu, args.model) # Initialize imported module
    print("Creating Temporary Files Directory")
    print("----------------------------------")
    try:
        print("01. Preparation Phase")
        di.temp_dir() # Create Temporary Directory
        di.vid_frames()
        print("02. Difference Generation")
        di.dif_frames()
        print("03. Post Processing")
        di.gen_video()
        print("    02. Removing Temporary Directory")
        di.temp_dir()
    except Exception as e:
        di.temp_dir()
        print("----------------------------------")
        print("Program Failed on Runtime due to Exception:")
        print(e)
    except KeyboardInterrupt:
        di.temp_dir()
        sys.exit(130)

if __name__ == '__main__':
    main()
