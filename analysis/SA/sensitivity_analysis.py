import time
import logging
import copy
import sys
import pandas as pd
import numpy as np
import pickle as pkl

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from numpy.random import Generator, MT19937

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

def main(
	period: str,
	basins: Optional[list] = None
):

	# Load the config.
	run_dir_path = Path("/home/wuhlmann/BA/repo/lamah-ce_lstm/models/q_pred_landcover_0903_135428")
	epoch = 16
	logger.info(f"Conducting SA for {run_dir_path} at epoch {epoch}. ")
	cfg = Config(run_dir_path / "config.yml")

	# Set up tester and get the baseline NSE, with no attributes altered.
	tester = get_tester(cfg=cfg, run_dir=run_dir_path, period=period, init_model=True)

	# If in training phase use basins splits to reduce computational load.
	if basins:
		tester.basins = basins
	
	logger.info(f"Tester initialized for {tester.period} period with basins {tester.basins}")
	logger.info("Conducting foward pass to calculate baseline NSE...")
	raw_results = tester.evaluate(epoch=epoch, save_results=False, metrics=["NSE"])

	# Get basin ids to iterate over.
	basin_ids_list = list(tester.cached_datasets.keys())
	
	baseline_nse_values = np.array([raw_results[id]["1D"]["NSE"] for id in basin_ids_list])
	
	# Instantiate custom RNG.
	rng = Generator(MT19937(seed=1277))

	# Sample 5 noise amounts per noise range, then created negative copy and run analysis.  
	noise_ranges = [0, 0.01, 0.1, 0.25, 0.5, 1]	

	noise_amounts = []

	for i in range(len(noise_ranges)-1): 
		
		base_noise_amounts = list(rng.uniform(low=noise_ranges[i],high=noise_ranges[i+1],size=2))

		noise_amounts.extend(list((map(lambda x: -x, base_noise_amounts))) + base_noise_amounts)

	# sort from - to + 
	noise_amounts.sort()

	num_basins = len(basin_ids_list)
	num_attr = len(cfg.static_attributes)
	noise_levels = len(noise_amounts)

	logger.info(f"Testing with the following {len(noise_amounts)} noise amounts:\n{noise_amounts}")

	# 3D array with dim basin x attribute x noise_level, to hold raw numeric values
	raw_array = np.zeros([num_basins, num_attr, noise_levels])

	for attr_id in range(num_attr):

		logger.info(f"Altering {tester.cfg.static_attributes[attr_id]} [{attr_id+1}/{num_attr}] ...")

		for noise_id in range(noise_levels): 

			logger.info(f"Appyling {noise_amounts[noise_id]} noise. ")

			# restore original attributes values in tester
			tester_copy = copy.deepcopy(tester)

			# add noise to the attribute in every catchment
			for id in basin_ids_list:	
				tester_copy.cached_datasets[id]._attributes[id][attr_id] += noise_amounts[noise_id]

			# run evaluation for the currrent noise level
			tester_result_dict = tester_copy.evaluate(epoch=epoch, save_results=False, metrics=["NSE"])
			nse_values = np.array([tester_result_dict[id]["1D"]["NSE"] for id in basin_ids_list])

			# write NSE for all basins, for the current attribute and noise level as difference with positive values meaning improvement compared to the baseline	
			raw_array[:, attr_id, noise_id] = nse_values - baseline_nse_values

	# Transfrom the results array into a dict of dataframes, so the basin id can be assigned and sorting is enable trough pandas.
	results_dict = {}

	for i in range(len(basin_ids_list)): 
		
		# Index for testing static. 
		results_dict[basin_ids_list[i]] = pd.DataFrame(data=raw_array[i,:,:], index=cfg.static_attributes, columns=noise_amounts)

	# Save nse deltas.
	base_nse_path = Path("/home/wuhlmann/BA/data/processed_data/SA/NSE_deltas")
	
	if not basins: 
		out_path = base_nse_path / f"{run_dir_path.stem}_{tester.period}_nse.p"
	else: 
		out_path = base_nse_path / f"{run_dir_path.stem}_{tester.period}_{tester.basins[0]}_{tester.basins[-1]}_nse.p"

	try: 
		with open(out_path, "wb") as out: 
			pkl.dump(results_dict, out)	
		logger.info(f"Successfully saved NSE deltas at {out_path}")
	
	except Exception as e: 

		logger.error(f"Pickling of NSE deltas failed:\n{e}")

if __name__ == "__main__":

	logging.basicConfig(
		level=logging.INFO,              
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)
	logger = logging.getLogger(__name__)
	logging.getLogger("neuralhydrology").setLevel(logging.CRITICAL)

	start_time = time.time()

	period = sys.argv[1]

	# If basin list is passed, load in. 
	if len(sys.argv) > 2: 
		basin_ids_path = Path(sys.argv[2])

		with open(basin_ids_path, "r") as f:
			# Skip last one, to avoid searching for basin " ".
			basin_ids_list = f.read().split("\n")[:-1]
		
		main(period=period, basins=basin_ids_list)

	else: 

		main(period=period)

	elapsed = time.time() - start_time
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = elapsed % 60

	logger.info(f"Sensitivity analysis completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")