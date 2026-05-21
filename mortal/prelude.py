import sys
import logging
import warnings
import os
from pathlib import Path

torchinductor_cache_dir = Path(__file__).resolve().parent / 'local_logs' / 'torchinductor_cache'
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(torchinductor_cache_dir))
torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)

import torch
import numpy as np

sys.stdin.reconfigure(encoding='utf-8')

logging.basicConfig(
    stream = sys.stderr,
    level = logging.INFO,
    format = '%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)

warnings.simplefilter('ignore')

# "The given NumPy array is not writeable"
dummy = np.array([])
dummy.setflags(write=False)
torch.as_tensor(dummy)

# "distutils Version classes are deprecated"
import torch.utils.tensorboard

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

warnings.simplefilter('default')
