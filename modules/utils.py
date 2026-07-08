from ffmpeg import FFmpeg, Progress
import tempfile
from pathlib import Path
from typing import Optional
import time

def convert(fromfilepath, tofilepath):
    ffmpeg = (
        FFmpeg()
        .input(fromfilepath)
        .output(tofilepath)
    )
    ffmpeg.execute()

class Temp:
    dir = Path(tempfile.mkdtemp())

def add_video_loop(videofilepath:Path, audiofilepath:Path, outfilepath:Path, seconds:Optional[float]=None, extras:dict={}):
    ffmpeg = (
        FFmpeg()
        .input(videofilepath)
        .input(audiofilepath)
        .output(outfilepath, options={'shortest':None, 'map':['1:a:0','0:v:0'], **extras})
        .option('stream_loop', -1)
        .option('y')
    )

    #for key, value in extras.items(): ffmpeg = ffmpeg.option(key, value)
    
    starttime = time.monotonic()
    @ffmpeg.on("progress")
    def on_progress(progress: Progress):
        if seconds:
            done = (progress.time.seconds/seconds)
            if done>0.01:
                remaining = (time.monotonic() - starttime)*(1-done)/done
                print(f"\rProcessed {done:>6.2%} - estimated time remaining {remaining:>4.0f}s  ", end='')
            else:
                print(f"\rProcessed {done:>6.2%}\r", end='')
        else:
            print(f"\rProcessed {progress.frame} frames\r", end='')
            
    ffmpeg.execute()

#ffmpeg  -stream_loop -1 -i videofilepath -i audiofilepath -shortest -map 0:v:0 -map 1:a:0 -y outfilepath
