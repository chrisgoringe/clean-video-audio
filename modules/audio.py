import torch
from torchcodec.decoders import AudioDecoder # type: ignore
from torchcodec.encoders import AudioEncoder # type: ignore
from torchcodec import AudioSamples # type: ignore
from noisereduce import reduce_noise
from noisereduce.torchgate import TorchGate

from pathlib import Path

class Audio:
    def __init__(self, sourcefile:str|Path):
        samples:AudioSamples = AudioDecoder(sourcefile).get_all_samples()

        self.data:torch.Tensor = samples.data
        self.dtype:torch.dtype = self.data.dtype
        self.sample_rate:int   = samples.sample_rate
        self.duration:float    = samples.duration_seconds

    def save(self, filepath):
        AudioEncoder(self.data, sample_rate=self.sample_rate).to_file(filepath)

    def enhance(self, nr:float=0.0, target_peak:float=0.0, use_torch=False) -> 'Audio':
        if nr:
            if use_torch:
                tg = TorchGate(sr=self.sample_rate, nonstationary=True).to("cuda")
                self.data = tg(self.data.cuda()).cpu()
            else:
                y = self.data.numpy()
                y = reduce_noise(y=y, sr=self.sample_rate, stationary=False, prop_decrease=nr)
                self.data = torch.from_numpy(y).to(self.dtype)
        if target_peak: 
            peak = self.data.abs().max().clamp_min(1e-8)
            self.data = target_peak * self.data / peak
        return self

