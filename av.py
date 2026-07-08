from modules.audio import Audio
from modules.enhancer import AudioEnhancer
from modules.utils import convert, Temp, add_video_loop
from modules.timer import Timer
from pathlib import Path
import argparse, yaml

def add_suffix_if_missing(p:Path, suffix:str) -> Path: return p if p.suffix else Path(f"{p}{suffix}")
def set_suffix(p:Path, suffix:str) -> Path: return p.parent / f"{p.stem}{suffix}"

def try_suffixes(p:Path, suffixes:list[str]) -> Path:
    if p.exists(): return p
    for s in suffixes:
        if (q := set_suffix(p, s)).exists(): return q
    raise FileNotFoundError(p)

def enhance_audio(audio_source:Path, savepath:Path, target_loudness:float) -> float: # length in seconds
    savepath = add_suffix_if_missing(savepath, '.wav')
    if savepath.exists(): raise FileExistsError(savepath)

    if not audio_source.exists():
        fp = try_suffixes(audio_source, ['.wav', '.mp3', 'mp4']) 
        if fp is None: raise FileNotFoundError(audio_source)
        audio_source = fp

    if not audio_source.suffix=='.wav':
        newpath = Temp.dir / 'temp.wav'
        with Timer("Converting to wav"): convert(audio_source, newpath)
        audio_source = newpath

    with Timer("Loading wav"): audio = Audio.from_file(audio_source)
    enhancer = AudioEnhancer()
    enhancer.enhance_audio(audio)
    enhancer.normalise_loudness(audio, target=target_loudness)

    if not savepath.suffix == '.wav':
        newpath = Temp.dir / 'temp.wav'
        with Timer("Saving as wav"): audio.save(newpath)
        with Timer("Converting from wav"): convert(newpath, savepath)
    else:
        with Timer("Saving as wav"): audio.save(savepath)

    return audio.samples / audio.samplerate

def patchvideo(audio_source:Path, savepath:Path, background_video:Path, do_audio_enhance:bool=True, target_loudness:float=-20):
    audio_source     = try_suffixes(audio_source, ['.wav', '.mp3']) 
    background_video = try_suffixes(background_video, ['.wav', '.mp3'])   

    savepath         = add_suffix_if_missing(savepath, '.wav')
    if savepath.exists(): raise FileExistsError(savepath)  

    if do_audio_enhance:
        enhanced_audio = Temp.dir / 'enhanced.wav' 
        n_seconds = enhance_audio(audio_source, enhanced_audio, target_loudness=-target_loudness)
    else:
        enhanced_audio = audio_source
        n_seconds = 0

    with Timer("Adding audio to video"):
        add_video_loop(
            videofilepath = background_video,
            audiofilepath = enhanced_audio,
            outfilepath   = savepath,
            seconds       = n_seconds,
            extras        = {'b:v':'500k'}
        )

DEFAULT_LOUDNESS = -20.0

def main(args:list[str]|None=None):
    a = argparse.ArgumentParser()
    a.add_argument('-i', '--in',    dest='i', type=Path, required=True, help='Source audio file')
    a.add_argument('-o', '--out',   dest='o', type=Path, required=True, help='output file')
    a.add_argument('-v', '--video', dest='v', type=Path, help='source video file')
    a.add_argument('--root',  type=Path, default='.', help=f'base directory (or shortcut) for input and output (default "downloads", shortcuts are {[k for k in paths]})')
    a.add_argument('--vroot', type=Path, default='.', help=f'base directory (or shortcut) for video (default "bg", shortcuts are {[k for k in paths]})')
    a.add_argument('--no_enhance', action='store_true', help="skip audio enhancement")
    a.add_argument('--loudness', type=float, default=DEFAULT_LOUDNESS, help=f"Target loudness (dB), default {DEFAULT_LOUDNESS}")
    a.add_argument('--action', choices=['auto', 'audio', 'video'], default='auto', help="")
    
    try:    arguments = a.parse_args(args)
    except: return print(HELP)

    if arguments.root  in paths: arguments.root  = paths[arguments.root ]
    if arguments.vroot in paths: arguments.vroot = paths[arguments.vroot]

    if arguments.action == 'auto':
        if arguments.o.suffix=='.mp3' or arguments.o.suffix=='.wav': arguments.action = 'audio'
        elif arguments.o.suffix=='.mp4': arguments.action = 'video'
        else:
            print("Couldn't work out action from --i and --o")
            return print(HELP)
        print(f"Setting action to {arguments.action}")

    for k,v in vars(arguments).items(): print(f'{k:>20} = {v}')

    if arguments.action == 'audio':
        enhance_audio(
            audio_source    = arguments.root / arguments.i, 
            savepath        = arguments.root / arguments.o, 
            target_loudness = arguments.loudness
        )
    elif arguments.action == 'video':
        patchvideo(
            audio_source     = arguments.root  / arguments.i,
            savepath         = arguments.root  / arguments.o,
            background_video = arguments.vroot / arguments.v,
            do_audio_enhance = not arguments.no_enhance,
            target_loudness  = arguments.loudness
        )

paths:dict[Path,Path] = {}
try:
    with open('shortcuts.yaml','r') as fh: 
        if (data:=yaml.safe_load(fh)): paths = { Path(k):Path(v) for k,v in data.items() }
except Exception as e:
    print(f"{e} when trying to load shortcuts")   

with open('README.md') as fh:
    HELP = (
        fh.read() + "\n" +
        (("Shortcuts defined:\n" + "\n".join( f"{str(k):>10} -> {str(v)}" for k,v in paths.items()))
            if paths else "None defined")
        )

if __name__=='__main__':
    main( )
    