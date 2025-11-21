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

def add_video_loop(videofilepath:Path, audiofilepath:Path, outfilepath:Path, seconds:Optional[float]=None):
    ffmpeg = (
        FFmpeg()
        .input(videofilepath)
        .input(audiofilepath)
        .output(outfilepath, options={'shortest':None, 'map':['1:a:0','0:v:0']})
        .option('stream_loop', -1)
        .option('y')
    )
    
    if seconds:
        starttime = time.monotonic()
        @ffmpeg.on("progress")
        def on_progress(progress: Progress):
            done = progress.time.seconds/seconds
            elapsed = time.monotonic() - starttime
            if done>0.01:
                remaining = elapsed*(1-done)/done
                print(f"\rProcessed {100*done:>6.2f}% - estimated time remaining {remaining:>4.0f}s  ", end='')
            else:
                print(f"\rProcessed {100*done:>6.2f}%\r", end='')
            

    @ffmpeg.on("completed")
    def on_completed():
        print("\r"+" "*60, end='')
    
    ffmpeg.execute()

#ffmpeg  -stream_loop -1 -i videofilepath -i audiofilepath -shortest -map 0:v:0 -map 1:a:0 -y outfilepath
