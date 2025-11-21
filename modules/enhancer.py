import torch, tempfile
from df import init_df, enhance
from .timer import Timer
import torchaudio
from .audio import Audio

class AudioEnhancer:
    def __init__(self):
        self.enhance_model, self.df_stats, _ = init_df()
        self.temp_dir = tempfile.mkdtemp()

    def enhance_chunk(self, chunk:torch.Tensor) -> torch.Tensor:
        return enhance(self.enhance_model, self.df_stats, chunk)
    
    def enhance_audio(self, audio:Audio, max_chunk=120):
        with Timer(f"Cleaning in {max_chunk}s chunks"):
            chunk_size = max_chunk*audio.samplerate
            chunks = 1 + ((audio.samples-1)//chunk_size)
            for i in range(0, chunks):
                with Timer(f"Cleaning chunk {i+1}/{chunks}"):
                    audio.wav[:, i*chunk_size:(i+1)*chunk_size] = self.enhance_chunk(audio.wav[:, i*chunk_size:(i+1)*chunk_size])
    
    def normalise_loudness(self, audio:Audio, target:float=-20):
        with Timer(f"Normalising to {target}"):
            if audio.wav.pow(2).mean().sqrt().item() < 2.e-3: return audio
            transform = torchaudio.transforms.Loudness(audio.samplerate)
            db = transform(audio.wav).mean().item()
            delta = target - db
            audio.wav = audio.wav * 10**(delta/20)
