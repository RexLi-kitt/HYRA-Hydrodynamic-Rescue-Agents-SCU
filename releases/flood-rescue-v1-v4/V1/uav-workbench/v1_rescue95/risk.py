from pathlib import Path
import numpy as np
from .deep import Deep

class Risk(Deep):
    def __init__(self):
        super().__init__(depth=14,width=32,commit=6)
        with np.load(Path(__file__).with_name('train-prior.npz')) as f:
            self.prior=f['q'].astype(float).reshape(31,-1)

