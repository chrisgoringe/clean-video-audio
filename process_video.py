
from typing import Optional

import torchaudio, torch, torchvision
from df import init_df, enhance
import whisper

from timer import Timer

#torch.set_default_dtype(torch.half)

x = torchaudio.utils.ffmpeg_utils.get_audio_encoders()
pass

class VideoFile:
    enhance_model = None
    df_stats = None
    whisper_model = None

    def __init__(self, path, start_s:float=0, end_s:Optional[float]=None):
        self.audio_reader = torchvision.io.VideoReader(src=path, stream='audio') 
        self.metadata = self.audio_reader.get_metadata()
        self.audio = None
        self.audio_file = None
        pass

    def load_audio(self):
        frames = [frame['data'] for frame in self.audio_reader]
        self.audio = torch.cat(frames)
        pass

    @property
    def video_fps(self) -> float:
        return self.metadata['video']['fps'][0]

    @property
    def audio_sample_rate(self) -> int:
        return int(self.metadata['audio']['framerate'][0])
    
    @property
    def audio_length(self) -> float:
        return self.metadata['audio']['duration'][0]
    
    @property
    def n_audio_samples(self)->int:
        return self.audio.shape[0]

    def clean_audio(self, chunk_size=2, normalise=True, normalise_target=-12):
        '''
        chunk_size in seconds. 
        If normalise it True (the default), normalise loudness to normalise_target dB (default -12)
        '''
        if VideoFile.enhance_model is None:
            VideoFile.enhance_model, VideoFile.df_stats, _ = init_df()
            #VideoFile.enhance_model.to(torch.half)
            VideoFile.enhance_model.requires_grad_(False)

        chunk_size = int(chunk_size * self.audio_sample_rate)
        for i in range(0, self.n_audio_samples, chunk_size):
            with Timer(f"Cleaning chunk {i//chunk_size+1}/{self.n_audio_samples//chunk_size+1}"):
                chunk = self.audio[i:i+chunk_size, :]#.to(torch.half)
                self.audio[i:i+chunk_size, :] = enhance(VideoFile.enhance_model, VideoFile.df_stats, chunk)#.to(torch.float)
        if normalise:
            with Timer("Normalising loudness"):
                self.normalise_loudness(normalise_target)

    def save_audio(self, path:str):
        torchaudio.save(uri=path, src=self.audio, sample_rate=self.audio_sample_rate, format='wav')
        self.audio_file = path

    def normalise_loudness(self, target:int=-12):
        if self.audio.pow(2).mean().sqrt().item() < 2.e-3: return
        db = torchaudio.transforms.Loudness(self.audio_sample_rate)(self.audio).mean().item()
        delta = target - db
        self.audio = self.audio * 10**(delta/20)

    def transcribe_audio(self):
        if VideoFile.whisper_model is None:
            VideoFile.whisper_model = whisper.load_model("turbo")
        if self.audio_file is None:
            self.save_audio('tempfile.wav')
        results = VideoFile.whisper_model.transcribe(self.audio_file, language='en')
        pass
    

filepath = r"C:\Users\chris\Dropbox\Family Stuff\Chris\Pastoral Supervision Course 2025\2025 Semester 1\Recording of 2025-03-25\video1916462910.mp4"
v = VideoFile(filepath)
v.load_audio()
#v.clean_audio()
v.transcribe_audio()
