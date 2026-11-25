from ffmpeg import FFmpeg, Progress
from pathlib import Path
from typing import Optional
import time

class Reporter:
    def __init__(self, seconds:float|None):
        self.seconds = seconds

    def start(self, args:list[str]):
        self.starttime = time.monotonic()

    def report(self, progress:Progress):
        elapsed = (time.monotonic() - self.starttime)
        print(f"\r{elapsed:>6.1}s ", end='')
        if self.seconds:
            done = (progress.time.seconds/self.seconds)
            if done>0.01:
                print(f"Processed {done:>6.2%} - ~{elapsed*(1-done)/done:>4.0f}s to go ", end='')
            else:
                print(f"Processed {done:>6.2%}", end='')
        else:
            print(f"\rProcessed {progress.frame} frames", end='')

    def stderr(self, line:str): pass
    def completed(self): pass
    def terminated(self): pass

def ffmpeg_executor(ffmpeg:FFmpeg, **kwargs) -> FFmpeg:
    reporter = Reporter(**kwargs)

    @ffmpeg.on("start")
    def on_start(arguments:list[str]):   reporter.start(arguments)

    @ffmpeg.on("progress")
    def on_progress(progress: Progress): reporter.report(progress)

    @ffmpeg.on("stderr")
    def on_stderr(line:str): reporter.stderr(line)

    @ffmpeg.on("completed")
    def on_completed(): reporter.completed()
    
    @ffmpeg.on("terminated")
    def on_terminated(): reporter.terminated()

    return ffmpeg


def convert(fromfilepath, tofilepath):
    ffmpeg_executor(
        FFmpeg()
        .input(fromfilepath)
        .output(tofilepath)
    ).execute()


def add_video_loop(videofilepath:Path, audiofilepath:Path, outfilepath:Path, seconds:Optional[float]=None, extras:dict={}):
#ffmpeg  -stream_loop -1 -i videofilepath -i audiofilepath -shortest -map 0:v:0 -map 1:a:0 -y outfilepath
    ffmpeg_executor(
        FFmpeg()
        .input(videofilepath)
        .input(audiofilepath)
        .output(outfilepath, options={'shortest':None, 'map':['1:a:0','0:v:0'], **extras})
        .option('stream_loop', -1)
        .option('y'),
        seconds = seconds
    ).execute()