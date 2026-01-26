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

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

def main(
	period: str,
	basins: Optional[list] = None
):

	# Load the config.
	run_dir_path = Path("/home/wuhlmann/BA/test_runs/runs/full_q_512_3011_185525")
	cfg = Config(run_dir_path / "config.yml")

	# Set up tester and get the baseline NSE, with no attributes altered.
	tester = get_tester(cfg=cfg, run_dir=run_dir_path, period=period, init_model=True)

	# If in training phase use basins splits to reduce computational load.
	if basins:
		tester.basins = basins
		
	logger.info(f"Tester initialized for {tester.period} period with basins {tester.basins}")
	logger.info("Conducting foward pass to calculate baseline NSE...")
	raw_results = tester.evaluate(save_results=False, metrics=["NSE"])

	# Get basin ids to iterate over.
	basin_ids_list = list(tester.cached_datasets.keys())
	
	baseline_nse_values = np.array([raw_results[id]["1D"]["NSE"] for id in basin_ids_list])
	
	# Conduct sensitivity analysis. Dont do absolute noise, use relative instead. Intervall von [-10%, 10%] variance as sens measure. 
	noise_amounts = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]

	num_basins = len(basin_ids_list)
	num_attr = len(cfg.static_attributes)
	noise_levels = len(noise_amounts)

	# 3D array with dim basin x attribute x noise_level, to hold raw numeric values
	raw_array = np.zeros([num_basins, num_attr, noise_levels])

	for attr_id in range(num_attr):

		logger.info(f"Altering {tester.cfg.static_attributes[attr_id]} [{attr_id+1}/{num_attr}] ...")

		for noise_id in range(noise_levels): 

			# restore original attributes values in tester
			tester_copy = copy.deepcopy(tester)

			# add noise to the attribute in every catchment
			for id in basin_ids_list:	
				tester_copy.cached_datasets[id]._attributes[id][attr_id] += noise_amounts[noise_id]

			# run evaluation for the currrent noise level
			tester_result_dict = tester_copy.evaluate(save_results=False, metrics=["NSE"])
			nse_values = np.array([tester_result_dict[id]["1D"]["NSE"] for id in basin_ids_list])

			# write NSE for all basins, for the current attribute and noise level	
			raw_array[:, attr_id, noise_id] = abs(nse_values - baseline_nse_values)

	# Transfrom the results array into a dict of dataframes, so the basin id can be assigned and sorting is enable trough pandas.
	results_dict = {}

	for i in range(len(basin_ids_list)): 
		
		# Index for testing static. 
		results_dict[basin_ids_list[i]] = pd.DataFrame(data=raw_array[i,:,:], index=cfg.static_attributes, columns=noise_amounts)

	# Save raw nse deltas in case a different weighting scheme is needed.
	base_raw_nse_path = Path("/home/wuhlmann/BA/data/processed_data/SA/raw_NSE")
	
	if not basins: 
		raw_out_path = base_raw_nse_path / f"{run_dir_path.stem}_{tester.period}_nse.p"
	else: 
		raw_out_path = base_raw_nse_path / f"{run_dir_path.stem}_{tester.period}_{tester.basins[0]}_{tester.basins[-1]}_nse.p"

	try: 
		with open(raw_out_path, "wb") as out: 
			pkl.dump(results_dict, out)	
		logger.info(f"Successfully saved raw NSE deltas at {raw_out_path}")
	
	except Exception as e: 

		logger.error(f"Pickling of raw NSE deltas failed:\n{e}")

	domain_ranks_dict = {}

	# Calculate the attribute sensitivity ranking per basin. 
	for id in results_dict.keys():

		# Get df.
		basin_df = results_dict[id]

		# Create df to collect the ranks per noise level.
		basin_attr_ranks = pd.DataFrame(data=[0]*len(basin_df.index), index=basin_df.index, columns=["rank"], dtype="float")

		attr_weights = [0.1, 0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25, 0.1]

		# Iterate over the noise level in the columns. 
		for col, weight in zip(basin_df.columns, attr_weights):
			
			# Sort the df by that noise level, with higher values in the NSE deltas signaling higher sensitivity.  
			order = list(basin_df.sort_values(by=col, ascending=False)[col].index)

			# Add the weighted rank of the attribute to the results dict. 
			for attr in basin_df.index: 

				basin_attr_ranks.loc[attr, "rank"] += (order.index(attr)) * weight

		# Divide through number of noise levels to get mean rank and sort the attributes by that. 
		mean_basin_attr_ranks = basin_attr_ranks.apply(lambda x: x/len(noise_amounts))
		mean_basin_attr_ranks.sort_values(by="rank", ascending=True, inplace=True)

		# Attach mean_attr_ranks for each basin. 
		domain_ranks_dict[id] = mean_basin_attr_ranks

	domain_ranks_base_path = Path("/home/wuhlmann/BA/data/processed_data/SA/ranks")	

	if not basins:
		domain_ranks_out_path =  domain_ranks_base_path/ f"{run_dir_path.stem}_{tester.period}_ranks.p"
	else:
		domain_ranks_out_path =  domain_ranks_base_path/ f"{run_dir_path.stem}_{tester.period}_{tester.basins[0]}_{tester.basins[-1]}_ranks.p"

	try: 
		with open(domain_ranks_out_path, "wb") as out: 
			pkl.dump(domain_ranks_dict, out)	
		logger.info(f"Successfully saved weigthed mean attribute ranks at {domain_ranks_out_path}")
	
	except Exception as e: 

		logger.error(f"Pickling of weighted mean attributed ranks failed:\n{e}")

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