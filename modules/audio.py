from dataclasses import dataclass
import torch, torchaudio

@dataclass
class Audio:
    wav:torch.Tensor
    samplerate:int

    @property
    def samples(self):
        return self.wav.shape[-1]
    
    @classmethod
    def from_file(cls, filepath) -> 'Audio':
        return cls(*torchaudio.load(str(filepath)))
    
    def save(self, filepath):
        torchaudio.save(str(filepath), self.wav, self.samplerate, channels_first=True)