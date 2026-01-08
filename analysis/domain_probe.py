# utils
import sys
import time
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm

#sk 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

def get_data_from_basin(
	ts_path: Path,
	target_var: str,
	basin_id: int,
	discharge_coverage: pd.DataFrame,
	cell_state_dict: dict,
):
	# normalize cell states, since they are not necessarily on the same scale
	cell_states = StandardScaler().fit_transform(cell_state_dict[str(basin_id)]["c_last"])

	ts = pd.read_table(ts_path / f"ID_{basin_id}.csv", header=0, sep=";")

	ts["date"] = pd.to_datetime(
				ts[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
			)
	ts.set_index("date", inplace=True)

	# drop all columns except for 2m_mean_temp
	temp = ts[[target_var]]

	# Cut temperature ts to match start and end date of the discharge ts, then remove the warmup period by only taking the len(c) days starting from the end. 
	temp = temp[discharge_coverage.loc[basin_id,"start_date"]:discharge_coverage.loc[basin_id, "end_date"]]
	temp = temp.iloc[-(len(cell_states)):]

	# Return array of dim num_days x num_cell_states+1(temp)
	basin_data_array = np.hstack([cell_states, temp.values])
	
	return basin_data_array

def fit_probe(
	dataset: dict
): 
	
	basin_ids = np.array(list(dataset.keys()))

	model = ElasticNet()

	num_folds = 5
	kfold = KFold(n_splits=num_folds)
	
	folds = kfold.split(basin_ids) #iterator wiht tuples[np.array, np.array] with the arrays being the train/test ids. 

	best_r2 = -100
	best_model_id = None
	model_list = []

	# Do 5 fold validation and let the best model predict (probe) on the entire timeseries.
	for i, f in tqdm(zip(list(range(num_folds)), folds), total=num_folds):
	
		train_ids = basin_ids[f[0]]
		test_ids = basin_ids[f[1]]

		training_set = np.vstack(itemgetter(*train_ids)(dataset))
		test_set = np.vstack(itemgetter(*test_ids)(dataset))

		model.fit(X=training_set[:,:-1], y=training_set[:,-1])
		pred = model.predict(X=test_set[:,:-1])

		# Collect all models.
		model_list.append(model)

		r2 = r2_score(test_set[:,-1], pred)

		if r2 > best_r2:
			best_r2 = r2
			best_model_id = i

	best_model = model_list[best_model_id]
	
	total_data = np.vstack(list(dataset.values()))

	logger.info(f"total data shape: {total_data.shape}")
	
	total_pred = best_model.predict(total_data[:,:-1])
	total_r2 = r2_score(total_data[:,-1], total_pred)
	total_mae = mean_absolute_error(total_data[:,-1], total_pred)

	return total_r2, total_mae, best_model

def main():

	cell_states_path = Path(sys.argv[1])
	target_var = str(sys.argv[2])
	
	logger.info(f"Setting up domain probe for {target_var} on {cell_states_path.stem} cell states.")

	with open(cell_states_path, "rb") as f: 
		cell_state_dict = pickle.load(f)
		
	cell_state_dict = dict(sorted(cell_state_dict.items(), key=lambda item: int(item[0])))

	# Get discharge coverage, to slice timeseries according to discharge timeseries.
	with open("/home/wuhlmann/BA/data/processed_data/eda/normal_periods.pkl", "rb") as f: 
		discharge_coverage = pickle.load(f)

	ts_path = Path("/home/wuhlmann/BA/data/raw_data/2_LamaH-CE_daily/B_basins_intermediate_all/2_timeseries/daily")

	probe_results_dict = defaultdict(dict)

	# Collect data from all basments. 
	dataset = {} 

	logger.info("Collecting training data...")

	for basin_id in tqdm(cell_state_dict.keys()):

		basin_data_array = get_data_from_basin(
			ts_path=ts_path,
			target_var=target_var,
			basin_id=int(basin_id),
			discharge_coverage=discharge_coverage,
			cell_state_dict=cell_state_dict
		)
		dataset[basin_id] = basin_data_array

	logger.info("Fitting probe...")

	# Do CV on different basins
	r2, mae, model = fit_probe(
		dataset=dataset
	)

	logger.info(f"Probe training results: R2 {r2:.2f} | MAE {mae:.2f}")

	probe_results_dict["train_results"]["R2"] = r2
	probe_results_dict["train_results"]["MAE"] = mae
	probe_results_dict["train_results"]["model"] = model 

	logger.info("Appyling probe to all basins.")
	for basin_id, data in tqdm(dataset.items()):

		pred = model.predict(data[:,:-1])
		basin_r2 = r2_score(data[:,-1], pred)
		basin_mae = mean_absolute_error(data[:,-1], pred)

		logger.info(f"Model performance for basin {basin_id}: R2 {basin_r2:.2f} | MAE {basin_mae:.2f}")

		probe_results_dict[basin_id]["R2"] = basin_r2
		probe_results_dict[basin_id]["MAE"] = basin_mae

	try: 
		out_path = Path("/home/wuhlmann/BA/data/processed_data/probing") / f"{cell_states_path.stem}_{target_var}_domain_probe.p"

		with open(out_path, "wb") as out: 
			pickle.dump(probe_results_dict, out)

		logger.info(f"Probing results successfully saved at: {out_path}")

	except Exception as e: 

		logger.error(f"Pickling of probing results failed:\n{e}")

	return


if __name__ == "__main__":

	logging.basicConfig(
        level=logging.INFO,              
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
	logger = logging.getLogger(__name__)

	start_time = time.time()

	main()

	elapsed = time.time() - start_time
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = elapsed % 60

	logger.info(f"Probing completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")