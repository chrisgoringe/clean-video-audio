from modules.audio import Audio
from modules.enhancer import AudioEnhancer
from modules.utils import convert, Temp, add_video_loop
from modules.timer import Timer
from pathlib import Path
import argparse

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



HELP = '''# Clean and normalised audio, and optionally add video background.

`python audio.py [action] --i input_file --o output_file [--b background_video_file]`

- input_file can be any media file that ffmpeg can read (including a video file)
- If extensions are omitted for input they will be guessed (.wav, .mp3, .mp4 for audio, .mp4 for video)
- If extensions are omitted for output .wav or .mp4 will be used.

## auto
action will be guessed from extension of output_file (.wav or .mp3 -> audio, .mp4 -> videopatch)

## audio 
read an audiofile, clean and normalise it, and save the output
`python audio.py [audio] --i audio_file --o output_audiofile`

## video
read an audiofile, clean and normalise it, add a video (with looping), and save the output. 
`python audio.py [video] --i audio_file --v video_file --o output_videofile`

If audio_file == video_file, this cleans the audio of an existing video

## Other options
- `--no_enhance` : with patchvideo, skip the audio enhance step
- `--loudness`   : target loudness (in dB) for normalisation (default -20)
- `--root`       : root directory for --i, --v, --o
'''

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--i', type=Path, required=True, help='Source audio file')
    a.add_argument('--o', type=Path, required=True, help='output file')
    a.add_argument('--v', type=Path, default='Stained_Glass_Scene_14_hd_1080.mp4', help='background video loop')
    a.add_argument('--root', type=Path, default=default_path, help=f'base directory for input and output (default {default_path})')
    a.add_argument('--no_enhance', action='store_true', help="skip audio enhancement")
    a.add_argument('--loudness', type=float, default=-20.0, help="Target loudness (dB), default -20")
    a.add_argument('action', choices=['auto', 'audio', 'video'], default='auto', help="")
    
    try:    arguments = a.parse_args()
    except: return print(HELP)

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
            audio_source     = arguments.root / arguments.i,
            savepath         = arguments.root / arguments.o,
            background_video = arguments.root / arguments.v,
            do_audio_enhance = not arguments.no_enhance,
            target_loudness  = arguments.loudness
        )

default_path = Path(r"C:\Users\chris\Dropbox\Roseville\YA Study\Dear Kim")

if __name__=='__main__':
    main()
    