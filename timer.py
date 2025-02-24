import time, os
class Timer:
    depth = 0
    filepath = None

    def __init__(self, label:str):
        self.label = label.strip('- ')
        
    def __enter__(self):
        Timer.depth += 1
        print("-"*Timer.depth*3+f" {self.label}")
        self.startat = time.monotonic()

    def __exit__(self, *args, **kwargs):
        tt = time.monotonic() - self.startat
        string = "-"*Timer.depth*3+f" {self.label} took {tt:>6.2f}s"
        print(string)
        if self.filepath: 
            with open(file=self.filepath, mode='a') as f: print(string, file=f, flush=True)
        Timer.depth -= 1

    @classmethod
    def timer_log_file(cls, filepath, restart=True):
        cls.filepath = filepath
        if restart and os.path.exists(cls.filepath): os.remove(cls.filepath)