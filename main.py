import argparse
import multiprocessing
import os.path
import sys
import termcolor
import traceback
from src.differ import Difference


def parse() -> argparse.Namespace:
    """Function for parsing CLI arguments"""
    sr_models = [
        os.path.splitext(os.path.basename(x))[0] for x in os.listdir(
            os.path.join(
                os.path.dirname(__file__),
                "weights"
            )
        ) if ".pth" in x
    ]
    sr_models.sort()

    if not len(sr_models):
        sr_models.append("Directory Empty!")

    parser = argparse.ArgumentParser(description="    " + termcolor.colored(" _____       ", "light_blue", attrs=["bold"]) + termcolor.colored("__  __       ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|  __ \     ", "light_blue", attrs=["bold"]) + termcolor.colored("|  \/  |      ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("| |__) |   _", "light_blue", attrs=["bold"]) + termcolor.colored("| \  / | ___  ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|  ___/ | | |", "light_blue", attrs=["bold"]) + termcolor.colored(" |\/| |/ _ \ ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("| |   | |_| |", "light_blue", attrs=["bold"]) + termcolor.colored(" |  | | (_) |", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|_|    \__, |", "light_blue", attrs=["bold"]) + termcolor.colored("_|  |_|\___/ ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("        __/ |", "light_blue", attrs=["bold"]) + termcolor.colored("             ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("       |___/ ", "light_blue", attrs=["bold"]) + termcolor.colored("             ", "light_green", attrs=["bold"]) + "\n\n" + \
    "   " + termcolor.colored("Python ", "light_blue", attrs=["bold"])  + termcolor.colored("Motion ", "light_green", attrs=["bold"]) + \
    termcolor.colored("Visualizer CLI", "white", attrs=["bold"]), formatter_class=argparse.RawTextHelpFormatter)

    required = parser.add_argument_group("Required", "Required Arguments")
    optional = parser.add_argument_group("Optional", "Optional Arguments")
    denoise = parser.add_argument_group("Denoising", description="Denoising Arguments")
    required.add_argument("-i", "--input", required=True,
                        help="""Relative path to the Input Video""", type=str)
    optional.add_argument("-o", "--output", required=False,
                        help="(Optional) Absolute path to output directory", type=str,metavar="PATH")
    optional.add_argument("-n", "--name", required=False, help="(Optional) Custom Filename for the video",
                        type=str)
    optional.add_argument("-f", "--offset", required=False, default=5,
                        help="(Optional) Number of Offset Frames [Default = 5]",type=int, choices=range(1, 50),
                        metavar="INT[1, 50]")
    optional.add_argument("-t", "--threads", required=False, default=2,
                        help="(Optional) Amount of threads to run the process on [Default = 2]",
                        type=int, choices=range(1, multiprocessing.cpu_count()),
                        metavar="CPU[1, %d]" % multiprocessing.cpu_count())
    optional.add_argument("-s", "--slow_motion", required=False, default=False,
                        help="""(Optional) Sets the FPS of the Output Video to half the original;
           Essentially creating a slow-motion of the original without interpolation""", action="store_true")
    denoise.add_argument("-c", "--cpu", required=False, default=False,
                         help="""(Optional) Denoising step by default runs on CUDA Acceleration (if Nvidia GPU Available);
           Setting this makes it run on CPU even if GPU is Available""", action="store_true")
    denoise.add_argument("-m", "--model", required=False, default="knn",
                        help="""(Optional) Model to use when denoising via GPU [Default = knn];
           SR - Super Resolution (ESRGAN, SwinIR, ...), 
           Needs a PyTorch weight file (.pth) in ./weights/ to be active.""",
                        choices=["knn", "nlm", "sr"])
    denoise.add_argument("-sr", "--super_resolution", required=False,
                        help=f'''(Optional) Choosing Weights for SR (if available)
           [Default = {sr_models[0]}]''',
                        choices=sr_models, default=sr_models[0])

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main() -> None:
    args = parse() # Parse CLI Arguments
    print("    " + termcolor.colored(" _____       ", "light_blue", attrs=["bold"]) + termcolor.colored("__  __       ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|  __ \     ", "light_blue", attrs=["bold"]) + termcolor.colored("|  \/  |      ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("| |__) |   _", "light_blue", attrs=["bold"]) + termcolor.colored("| \  / | ___  ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|  ___/ | | |", "light_blue", attrs=["bold"]) + termcolor.colored(" |\/| |/ _ \ ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("| |   | |_| |", "light_blue", attrs=["bold"]) + termcolor.colored(" |  | | (_) |", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("|_|    \__, |", "light_blue", attrs=["bold"]) + termcolor.colored("_|  |_|\___/ ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("        __/ |", "light_blue", attrs=["bold"]) + termcolor.colored("             ", "light_green", attrs=["bold"]) + "\n" + \
    "    " + termcolor.colored("       |___/ ", "light_blue", attrs=["bold"]) + termcolor.colored("             ", "light_green", attrs=["bold"]) + "\n\n" + \
    "   " + termcolor.colored("Python ", "light_blue", attrs=["bold"])  + termcolor.colored("Motion ", "light_green", attrs=["bold"]) + \
    termcolor.colored("Visualizer CLI", "white", attrs=["bold"]) + "\n")
    print("Initializing")
    di = Difference(args.input, args.output, args.name, args.offset, args.threads,
                    args.slow_motion, args.cpu, args.model, args.super_resolution) # Initialize imported module
    print("Creating Temporary Files Directory")
    print("----------------------------------")
    try:
        print("01. Preparation Phase")
        di.temp_dir() # Create Temporary Directory
        di.vid_frames()
        print("02. Difference Generation")
        di.dif_frames()
        print("03. Post Processing")
        output = di.gen_video()
        print("    02. Removing Temporary Directory")
        di.temp_dir()
        print("----------------------------------")
        print(termcolor.colored("Finished!", "light_green", attrs=["bold"]))
        print(f"Video rendered at {termcolor.colored(output, color='light_yellow')}")
    except Exception as e:
        di.temp_dir()
        print("----------------------------------")
        print(f">>> {termcolor.colored('Program Failed on Runtime due to Exception:', 'red', 'on_black', ['bold'])}")
        print(f">>> {termcolor.colored(e, on_color='on_black', attrs=['bold'])}")
        traceback.print_exc()
    except KeyboardInterrupt:
        print("----------------------------------")
        print(f">>> {termcolor.colored('KeyboardInterrupt', 'cyan', attrs=['bold'])}")
        print(">>> Removing Temporary Directory")
        di.temp_dir()
        sys.exit(130)

if __name__ == '__main__':
    main()
