from ffmpeg.ffmpeg import FFmpeg
import torchaudio
import os, tempfile
from df import init_df, enhance
from functools import partial
from modules.timer import Timer
import argparse, random

class Enhancer:
    enhance_model = None
    df_stats = None
    @classmethod
    def init(cls):
        if cls.enhance_model is None:
            cls.enhance_model, cls.df_stats, _ = init_df()
            cls.temp_dir = tempfile.mkdtemp()

class Temp:
    temp_dir = tempfile.mkdtemp()
    file = partial(os.path.join, temp_dir)

def normalise_loudness(wav, sample_rate, target=-12):
    if wav.pow(2).mean().sqrt().item() < 2.e-3: return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    db = transform(wav).mean().item()
    delta = target - db
    return wav * 10**(delta/20)

def extract_audio(videofilepath, audiofilepath=None):
    audiofilepath = audiofilepath or Temp.file(f"audio{random.randint(100000,999999)}.wav")
    if os.path.exists(audiofilepath): 
        print(f"**WARNING** File {audiofilepath} already exists - not doing a fresh extraction")
        return audiofilepath
    ffmpeg = (
        FFmpeg()
        .input(videofilepath)
        .output(audiofilepath)
    )
    ffmpeg.execute()
    return audiofilepath

def clean_audio(audiofilepath, chunk=120):
    saveaudiofilepath = f"{os.path.splitext(audiofilepath)[0]}-clean{os.path.splitext(audiofilepath)[1]}"
    wav, samplerate = torchaudio.load(audiofilepath)
    chunk_size = chunk * samplerate
    for i in range(0, wav.shape[1], chunk_size):
        with Timer(f"Cleaning chunk {i//chunk_size+1}/{wav.shape[1]//chunk_size+1}"):
            chunk = wav[:, i:i+chunk_size]
            wav[:, i:i+chunk_size] = enhance(Enhancer.enhance_model, Enhancer.df_stats, chunk)
    with Timer(f"Saving cleaned audio to {saveaudiofilepath}"):
        torchaudio.save(saveaudiofilepath, wav, samplerate, channels_first=True)
    return saveaudiofilepath

def replace_audio(videofilepath, audiofilepath, savevideofilepath=None):
    savevideofilepath = savevideofilepath or Temp.file('cleaned_video.mp4')

    if os.path.exists(savevideofilepath): 
        print(f"**WARNING** File {savevideofilepath} already exists - ", end='')
        i = 0
        while(os.path.exists(f"{os.path.splitext(savevideofilepath)[0]}-{i}{os.path.splitext(savevideofilepath)[1]}")): i = i + 1
        savevideofilepath = f"{os.path.splitext(savevideofilepath)[0]}-{i}{os.path.splitext(savevideofilepath)[1]}"
        print(f"saving as {savevideofilepath}")

    ffmpeg = (
        FFmpeg()
        .input(videofilepath)
        .input(audiofilepath)
        .output(savevideofilepath, vcodec='copy', acodec='aac', map=["0:v:0","1:a:0"])
    )
    ffmpeg.execute()
    return savevideofilepath

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, type=str, help='Input video file')
    parser.add_argument('--outfile', type=str, help='Output video file. If not specified, -clean will be appended to input file name')
    parser.add_argument('--directory', type=str, help='Optional root directory for video files')
    parser.add_argument('--keep', action='store_true', help='Keep temporary files')
    parser.add_argument('--chunk', type=int, default=120, help='Chunk size in seconds for processing audio')

    args = parser.parse_args()
    path = partial(os.path.join, args.directory) if args.directory else os.path.join

    videofile = path(args.infile)
    cleanvideofile = path(args.outfile) if args.outfile else f"{os.path.splitext(videofile)[0]}-clean{os.path.splitext(videofile)[1]}"

    if args.keep: Temp.file = partial(os.path.join, (args.directory or ".")) 

    with Timer(f"Cleaning {videofile} -> {cleanvideofile}"):
        with Timer('Extract audio'): wavfile = extract_audio(videofile)
        with Timer('Load model'):    Enhancer.init()
        with Timer('Clean audio'):   cleanwavfile = clean_audio(wavfile, chunk=args.chunk)
        with Timer('Replace audio'): replace_audio(videofile, cleanwavfile, cleanvideofile)