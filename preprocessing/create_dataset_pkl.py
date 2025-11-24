import sys 
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config

from pathlib import Path 

cfg_path = Path("/home/wuhlmann/BA/repo/lamah-ce_lstm/testing/test_128_on_q_filter.yml")
cfg = Config(cfg_path)

ds = get_dataset(cfg=cfg, is_train=True, period="train")

