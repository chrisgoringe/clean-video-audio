from modules.audio import Audio
from modules.enhancer import AudioEnhancer
from modules.utils import convert, Temp, add_video_loop
from modules.timer import Timer
from pathlib import Path
from typing import Optional
import argparse

def enhance(filepath:Path, savepath:Path, target_loudness:float, replace:bool=False) -> float: # length in seconds

    if savepath.exists():
        if replace: savepath.unlink()
        else: raise FileExistsError(savepath)

    if not filepath.suffix=='.wav':
        newpath = Temp.dir / 'temp.wav'
        with Timer("Converting to wav"): convert(filepath, newpath)
        filepath = newpath

    with Timer("Loading wav"): audio = Audio.from_file(filepath)
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

def run_audio_video_merge(audio_source:Path, outfilepath:Path, background_video:Path, enhanced_audio:Optional[Path]=None, enhance_audio:bool=True):
    enhanced_audio = (enhanced_audio or Temp.dir / 'enhanced.wav') if enhance_audio else None
    if enhanced_audio: 
        n_seconds = enhance(audio_source, enhanced_audio, target_loudness=-20)
    else:
        n_seconds = None
    with Timer("Adding audio to video"):
        add_video_loop(
            videofilepath = background_video,
            audiofilepath = enhanced_audio or audio_source,
            outfilepath   = outfilepath,
            seconds       = n_seconds,
            extras        = {'b:v':'500k'}
        )





backgrounds = Path(r'C:\Users\chris\OneDrive - Roseville Uniting Church\ruc.multimedia\Share Faith\Background Videos (Background videos)')
downloads   = Path(r'C:\Users\chris\Downloads')

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--i', type=Path, required=True, help='Source audio file')
    a.add_argument('--o', type=Path, required=True, help='output video file')
    a.add_argument('--root', type=Path, default=downloads, help=f'base directory for input and output (default {downloads})')
    a.add_argument('--broot', type=Path, default=backgrounds, help=f"base directory for backgrounds (default {backgrounds})")
    a.add_argument('--b', type=Path, default='Stained_Glass_Scene_14_hd_1080.mp4', help='background video loop')
    a.add_argument('--no_enhance', action='store_true', help="skip audio enhancement")
                   
    arguments = a.parse_args()

    for k,v in vars(arguments).items(): print(f'{k:>20} = {v}')
    def fix_suffix(p:Path, suffix:str) -> Path: return p if p.suffix else Path(f"{p}{suffix}")
    arguments.i = fix_suffix(arguments.i, '.mp3')
    arguments.o = fix_suffix(arguments.o, '.mp4')
    arguments.b = fix_suffix(arguments.b, '.mp4')

    run_audio_video_merge(
        audio_source     = arguments.root   / arguments.i,
        outfilepath      = arguments.root   / arguments.o,
        background_video = arguments.broot / arguments.b,
        enhance_audio    = not arguments.no_enhance
    )

if __name__=='__main__':
    main()
    