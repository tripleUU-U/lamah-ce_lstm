import time 
import logging 
import numpy as np
import torch
import pandas as pd
import pickle

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from neuralhydrology.modelzoo import EALSTM
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.utils.config import Config


def get_gradient(model: torch.nn.Module,
						loader: torch.utils.data.DataLoader) -> torch.Tensor:
	"""Updated from Kratzert et al. 2019, to work with EA-LSTM in NH package."""

	model.eval()
	grads = []

	# returns dict in NH
	for batch in loader:
		
		# mark static attributes for grdient computation
		batch["x_s"] = torch.autograd.Variable(batch["x_s"], requires_grad=True)

		model.zero_grad()
		
		pred = model(batch)

		# create mask to not compute gradients for days, where no discharge data exists, since these days are not in the set
		mask = ~torch.isnan(batch["y"])

		# compute gradients of prediction vector given the static attributes
		grad = torch.autograd.grad(pred["y_hat"],
								   batch["x_s"],
								   grad_outputs=mask.float(),
								   create_graph=False)
		
		# results in tuple with one tensor shaped batch size x num static attributes
		grads.append(grad[0][:,:].detach().cpu().numpy())

	return np.concatenate(grads, axis=0)


def main() -> None: 
	
	# load model version 
	run_dir_path = Path("/home/wuhlmann/BA/test_runs/runs/full_q_512_3011_185525")
	cfg = Config(run_dir_path/"config.yml")

	model = EALSTM(cfg=cfg)
	scaler = load_scaler(run_dir=run_dir_path)

	model_weights = torch.load("/home/wuhlmann/BA/test_runs/runs/q_pred_landcover_0903_135428/model_epoch016.pt", map_location="cuda:0")
	model.load_state_dict(model_weights)

	feature_ranking = {}

	period_pbar = tqdm(["train", "val", "test"], position=1)

	for period in period_pbar:
		
		logger.info(f"Conducting gradient analysis for {period} period.")

		# conduct analysis for every basin, to identify locally mosty important attributes.
		with open(f"/home/wuhlmann/BA/repo/lamah-ce_lstm/data/splits_incl_landcover/{period}_basin_ids.txt", "r") as test_basin_file: 
	
			basins = list(map(lambda x: x[:-1], test_basin_file.readlines()))

		basin_pbar = tqdm(basins, position=2)

		# rephrase for NH dataset
		if period == "val": 
			period = "validation"
		
		for basin in basin_pbar:
			
			ds_test = get_dataset(cfg=cfg, is_train=False, period=period, scaler=scaler, basin=basin)
			loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=0, collate_fn=ds_test.collate_fn)

			gradients = get_gradient(model, loader)

			# The first 365 days of gradients are nan, if the data starts at 1981, since the first predictions can be made in 1982.
			mean_abs_gradient = np.nanmean(np.abs(gradients), axis=0)

			# convert to pandas Series
			data = {}
			for name, value in zip(list(cfg.static_attributes), mean_abs_gradient):
				data[name] = value
			feature_ranking[basin] = pd.Series(data=data)

	out_path = Path(f"/home/wuhlmann/BA/data/processed_data/SA/gradients/{run_dir_path.stem}.p")

	with open(out_path, "wb") as out: 
		pickle.dump(feature_ranking, out)
	
	logger.info(f"Successfully saved NSE deltas at {out_path}")


if __name__ == "__main__":

	logging.basicConfig(
		level=logging.INFO,              
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)
	logger = logging.getLogger(__name__)
	logging.getLogger("neuralhydrology").setLevel(logging.CRITICAL)

	start_time = time.time()
 
	main()

	elapsed = time.time() - start_time
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = elapsed % 60

	logger.info(f"Sensitivity analysis completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")