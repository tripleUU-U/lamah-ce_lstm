import pickle as pkl

from tqdm import tqdm
from pathlib import Path

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

run_dir_path = Path("/home/wuhlmann/BA/repo/lamah-ce_lstm/models/q_pred_landcover_0903_135428")
epoch = 16
cfg = Config(run_dir_path / "config.yml")

input_gate_dict = {}

for period in tqdm(["train", "validation", "test"]):

	tester = get_tester(cfg=cfg, run_dir=run_dir_path, period=period, init_model=True)

	# tester only needs to run once, to get all i vectors
	tester.evaluate(epoch=epoch, save_results=False, metrics=["NSE"])

	for b in tester.basins: 

		# move tensor to same device as model
		x_s = tester.cached_datasets[b]._attributes[b].to("cuda:0")

		input_gate_dict[b] = tester.model.input_gate(x_s).detach().cpu().numpy()

with open("/home/wuhlmann/BA/data/processed_data/evaluation/input_gate.p", "wb") as f: 
	pkl.dump(input_gate_dict, f)
