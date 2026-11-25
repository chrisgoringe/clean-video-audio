
from typing import Optional
import os, json
import torchaudio, torch, torchvision
from df import init_df, enhance
import whisper
from modules.timer import Timer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoFile:
    enhance_model = None
    df_stats = None
    whisper_model = None

    def __init__(self, base, ext, start_s:float=0, end_s:Optional[float]=None):
        self.audio_reader = torchvision.io.VideoReader(src=f"{base}{ext}", stream='audio') 
        self.metadata = self.audio_reader.get_metadata()
        self.audio = None
        self.base = base
        self.start_s = start_s
        self.end_s = end_s
        m = f"{base}.clean_audio.wav"
        self.clean_audio_file = m if os.path.exists(m) else None
        self._audio_sample_rate = None

    def load_audio(self):
        if self.audio:
            print("Audio already loaded")
            return
        frames = [frame['data'] for frame in self.audio_reader]
        self.audio = torch.cat(frames, dim=0)
        self.audio = torch.mean(self.audio, dim=1) # to mono
        self.audio = self.audio[self.start_s*self.audio_sample_rate : (self.end_s*self.audio_sample_rate if self.end_s else None)]
        
        self.audio.unsqueeze_(0)

    @property
    def video_fps(self) -> float:
        return self.metadata['video']['fps'][0]

    @property
    def audio_sample_rate(self) -> int:
        if not self._audio_sample_rate: self._audio_sample_rate = int(self.metadata['audio']['framerate'][0])
        return self._audio_sample_rate
    
    @property
    def audio_length(self) -> float:
        return self.metadata['audio']['duration'][0]
    
    @property
    def n_audio_samples(self)->int:
        return self.audio.shape[1]

    def clean_audio(self, chunk_size=120, normalise=True, normalise_target=-12):
        '''
        chunk_size in seconds. 
        If normalise it True (the default), normalise loudness to normalise_target dB (default -12)
        '''
        if self.clean_audio_file: 
            print(f"{self.clean_audio_file} found, skipping clean")
            return
        self.load_audio()
        if VideoFile.enhance_model is None:
            VideoFile.enhance_model, VideoFile.df_stats, _ = init_df()
            VideoFile.enhance_model.requires_grad_(False)

        chunk_size = int(chunk_size * self.audio_sample_rate)
        for i in range(0, self.n_audio_samples, chunk_size):
            with Timer(f"Cleaning chunk {i//chunk_size+1}/{self.n_audio_samples//chunk_size+1}"):
                chunk = self.audio[:,i:i+chunk_size]
                self.audio[:,i:i+chunk_size] = enhance(VideoFile.enhance_model, VideoFile.df_stats, chunk)
        if normalise:
            with Timer("Normalising loudness"):
                self.normalise_loudness(normalise_target)

        VideoFile.enhance_model = None

    def save_audio(self, path:str):
        torchaudio.save(uri=path, src=self.audio, sample_rate=self.audio_sample_rate, format='wav')
        self.clean_audio_file = path

    def load_audiofile(self, path:str):
        self.audio, self._audio_sample_rate = torchaudio.load(uri=path)

    def normalise_loudness(self, target:int=-12):
        if self.audio.pow(2).mean().sqrt().item() < 2.e-3: return
        db = torchaudio.transforms.Loudness(self.audio_sample_rate)(self.audio).mean().item()
        delta = target - db
        self.audio = self.audio * 10**(delta/20)

    def transcribe_audio(self):
        if not self.clean_audio_file:
            self.save_audio(f"{self.base}.clean_audio.wav")
        if VideoFile.whisper_model is None:
            VideoFile.whisper_model = whisper.load_model("turbo")
            VideoFile.whisper_model.to(torch.float)
            VideoFile.whisper_model.to(DEVICE)
        results = VideoFile.whisper_model.transcribe(self.clean_audio_file, language='en')
        return results
    
def combine_segments(segments:list[dict]):
    txt = None
    last_end = 0
    gap = None
    for seg in segments:
        if txt is None: 
            gap = seg['start'] - last_end
            txt = seg['text']
        else:
            txt += " " + seg['text']
        if txt[-1]=='.':
            yield (gap, txt)
            txt = None
            gap = None
            last_end = seg['end']
    if txt is not None:
        yield (gap, txt)
    
def clean_and_transcribe(directory, file, para_gap:float, **kwargs):
    base = os.path.join(directory, os.path.splitext(file)[0])

    if os.path.exists(f"{base}.json"):
        with open(f"{base}.json", 'r') as fp:
            results = json.load(fp)
    else:
        v = VideoFile(base=base, ext=os.path.splitext(file)[1], **kwargs)
        with Timer("Clean"):      v.clean_audio()
        with Timer("Transcribe"): results = v.transcribe_audio()
        with open(f"{base}.json", 'w') as fp:
            print(json.dumps(results, indent=2), file=fp)

    with open(f"{base}.txt", 'w', encoding='UTF8') as fp: 
        for (gap, line) in combine_segments(results['segments']):
            if gap>para_gap: print("\n", file=fp)
            print(f"{gap:>.1f} {line}", file=fp, end="") 

if __name__=='__main__':
    directory = os.path.join(
        "/Users","chris","Downloads","2025-03-30"
    )
    file = "Roseville Uniting Church Livestream.mp4"
    start_m = 27
    start_s = 0
    end_m = 49
    end_s = 20

    para_gap = 1.7

    clean_and_transcribe(directory, file, start_s=start_s+60*start_m, end_s=end_s+60*end_m, para_gap=para_gap)