from modules.audio import Audio
from modules.enhancer import AudioEnhancer
from modules.utils import convert, Temp, add_video_loop
from modules.timer import Timer
from pathlib import Path

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

def main(audio_source:Path, outfilepath:Path, background_video:Path):
    enhanced_audio = Temp.dir / 'enhanced.wav'
    n_samples = enhance(audio_source, enhanced_audio, target_loudness=-20)
    with Timer("Adding audio to video"):
        add_video_loop(
            videofilepath = background_video,
            audiofilepath = enhanced_audio,
            outfilepath   = outfilepath,
            seconds = n_samples
        )

backgrounds = Path(r'C:\Users\chris\OneDrive - Roseville Uniting Church\ruc.multimedia\Share Faith\Background Videos (Background videos)')
downloads   = Path(r'C:\Users\chris\Downloads')

if __name__=='__main__':
    main(
        audio_source = downloads / '251123_NeilProudlove_Prayer.mp3',
        outfilepath  = downloads / '251123_NeilProudlove_Prayer.mp4',
        background_video = backgrounds / 'Dark Triangles.mp4'
    )

    