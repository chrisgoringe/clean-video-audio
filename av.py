from modules.audio import Audio
from modules.processing import add_video_loop
from modules.timer import Timer
from pathlib import Path
import argparse, yaml, tempfile

paths:dict[Path,Path] = {}
shortcuts_path = Path('shortcuts.yaml')
shortcuts_problem = ""
if shortcuts_path.exists():
    try:
        with open('shortcuts.yaml','r') as fh: 
            if (data:=yaml.safe_load(fh)): paths = { Path(k):Path(v) for k,v in data.items() }
    except Exception as e:
        shortcuts_problem = f"{e} when reading {shortcuts_path}"
else:
    shortcuts_problem = f"{shortcuts_path} not found" 

with open('README.md') as fh:
    HELP = fh.read() + "\n"
    if paths: 
        HELP += "Shortcuts defined:\n" + "\n".join( f"{str(k):>10} -> {str(v)}" for k,v in paths.items())
    else:
        HELP += f"No shortcuts loaded because {shortcuts_problem}."


def add_suffix_if_missing(p:Path, suffix:str) -> Path: return p if p.suffix else Path(f"{p}{suffix}")
def set_suffix(p:Path, suffix:str) -> Path: return p.parent / f"{p.stem}{suffix}"

def try_suffixes(p:Path, suffixes:list[str]) -> Path:
    if p.exists(): return p
    for s in suffixes:
        if (q := set_suffix(p, s)).exists(): return q
    raise FileNotFoundError(p)

def enhance_audio(audio_source:Path, savepath:Path, target_peak:float, nr:float) -> float: # length in seconds
    savepath = add_suffix_if_missing(savepath, '.wav')
    if savepath.exists(): raise FileExistsError(savepath)

    if not audio_source.exists():
        fp = try_suffixes(audio_source, ['.wav', '.mp3', 'mp4']) 
        if fp is None: raise FileNotFoundError(audio_source)
        audio_source = fp

    with Timer("Loading audio"): 
        audio = Audio(audio_source)
    
    with Timer("Enhancing audio"): 
        audio.enhance(nr, target_peak)

    with Timer("Saving audio"):
        audio.save(savepath)

    return audio.duration

def patchvideo(audio_source:Path, savepath:Path, background_video:Path, target_peak:float, nr:float):
    audio_source     = try_suffixes(audio_source, ['.wav', '.mp3']) 
    background_video = try_suffixes(background_video, ['.wav', '.mp3'])   

    savepath         = add_suffix_if_missing(savepath, '.wav')
    if savepath.exists(): raise FileExistsError(savepath)  

    audio = Audio(audio_source)
    audio.enhance(nr, target_peak)

    with tempfile.TemporaryDirectory() as tempdir:
        enhanced_audio = Path(tempdir) / 'enhanced.wav'
        audio.save(enhanced_audio)

        with Timer("Adding audio to video"):
            add_video_loop(
                videofilepath = background_video,
                audiofilepath = enhanced_audio,
                outfilepath   = savepath,
                seconds       = audio.duration,
                extras        = {'b:v':'500k'}
            )

DEFAULT_LOUDNESS = 0.95
DEFAULT_NR       = 0.8

# used for guessing...
AUDIO_SUFFICES = [ '.mp3', '.wav', '.m4a' ]
VIDEO_SUFFICES = [ '.mp4', '.mov', '.avi', '.wmv', '.mkv', '.mpg', '.3gp' ]

def parse_arguments(args:list[str]|None=None) -> dict:
    a = argparse.ArgumentParser()
    a.add_argument('-i', '--in',    dest='i', type=Path, help='Source audio file')
    a.add_argument('-o', '--out',   dest='o', type=Path, required=True, help='output file')
    a.add_argument('-v', '--video', dest='v', type=Path, help='source video file')
    a.add_argument('--root',  type=Path, default='.', help=f'base directory (or shortcut) for input and output (default "downloads", shortcuts are {[k for k in paths]})')
    a.add_argument('--vroot', type=Path, default='.', help=f'base directory (or shortcut) for video (default "bg", shortcuts are {[k for k in paths]})')
    a.add_argument('--nr', type=float, default=DEFAULT_NR, help=f"Noise reduction, zero to skip")
    a.add_argument('--loudness', type=float, default=DEFAULT_LOUDNESS, help=f"Target loudness (peak), 0 to skip")
    a.add_argument('--action', choices=['auto', 'audio', 'video'], default='auto', help="")
    
    try:    
        arguments = a.parse_args(args)

        if arguments.root  in paths: arguments.root  = paths[arguments.root ]
        if arguments.vroot in paths: arguments.vroot = paths[arguments.vroot]

        if arguments.action == 'auto':
            if    arguments.o.suffix.lower() in AUDIO_SUFFICES: arguments.action = 'audio'
            elif  arguments.o.suffix.lower() in VIDEO_SUFFICES: arguments.action = 'video'
            else: raise argparse.ArgumentError(argument=None, message="Couldn't work out action from --o")
            print(f"Setting action to {arguments.action}")

        
        return vars(arguments)
    except: 
        a.print_help()
        raise

def main(args:list[str]|None=None):
    try:
        arguments = parse_arguments(args)
        print("\n".join(f'{k:>20} = {v}' for k,v in arguments.items()))
    except argparse.ArgumentError as e:
        print(" ".join(e.args))
        print(HELP)
        return

    if arguments['action'] == 'audio':
        enhance_audio(
            audio_source  = arguments['root'] / arguments['i'], 
            savepath      = arguments['root'] / arguments['o'], 
            target_peak   = arguments['loudness'],
            nr            = arguments['nr']
        )
    elif arguments['action'] == 'video':
        patchvideo(
            audio_source     = arguments['root']  / arguments['i'],
            savepath         = arguments['root']  / arguments['o'],
            background_video = arguments['vroot'] / arguments['v'],
            target_peak      = arguments['loudness'],
            nr               = arguments['nr']
        )

if __name__=='__main__': main( )
    