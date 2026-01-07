# utils
import sys
import time
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from operator import itemgetter
from tqdm import tqdm

#sk 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

def get_data_from_basin(
	ts_path: Path,
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
	temp = ts[["2m_temp_mean"]]

	# Cut temperature ts to match start and end date of the discharge ts, then remove the warmup period by only taking the len(c) days starting from the end. 
	temp = temp[discharge_coverage.loc[basin_id,"start_date"]:discharge_coverage.loc[basin_id, "end_date"]]
	temp = temp.iloc[-(len(cell_states)):]

	basin_data_array = np.array([cell_states, temp.values])

	return basin_data_array

def fit_probe(
	dataset
): 

	model = ElasticNet()
	kfold = KFold(n_splits=5)
	folds = list(kfold.split(dataset.keys()))

	best_r2 = -100
	best_model_id = None
	model_list = []

	# Do 5 fold validation and let the best model predict (probe) on the entire timeseries.
	for i, f in zip([0, 1, 2, 3, 4], folds):

		train_ids = f[0]
		test_ids = f[1]

		training_set = np.hstack(itemgetter(*train_ids)(dataset))
		test_set = np.hstack(itemgetter(*test_ids)(dataset))

		model.fit(X=training_set[0,:] y=training_set[1,:])
		pred = model.predict(X=test_set[0,:])

		# Collect all models.
		model_list.append(model)

		r2 = r2_score(test_set[1,:], pred)

		if r2 > best_r2:
			best_r2 = r2
			best_model_id = i

	best_model = model_list[best_model_id]
	
	total_data = np.hstack(dataset.values())
	
	total_pred = best_model.predict(total_data[0,:])
	total_r2 = r2_score(total_data[1,:], total_pred)
	total_mae = mean_absolute_error(total_data[1,:], total_pred)

	# Only the coeffiecents of the model are needed, since it's not used again. 
	coef_list = best_model.coef_.tolist()

	return total_r2, total_mae, coef_list


def main():

	cell_states_path = Path(sys.argv[1])

	with open(cell_states_path, "rb") as f: 
		cell_state_dict = pickle.load(f)
		
	cell_state_dict = dict(sorted(cell_state_dict.items(), key=lambda item: int(item[0])))

	# Get discharge coverage, to slice timeseries according to discharge timeseries.
	with open("/home/wuhlmann/BA/data/processed_data/eda/normal_periods.pkl", "rb") as f: 
		discharge_coverage = pickle.load(f)

	ts_path = Path("/home/wuhlmann/BA/data/raw_data/2_LamaH-CE_daily/B_basins_intermediate_all/2_timeseries/daily")

	probe_results_dict = {}

	# Collect data from all basments. 
	dataset = {}

	for basin_id in tqdm(cell_state_dict.keys()):

		basin_data_array = get_data_from_basin(
			ts_path=ts_path,
			basin_id=basin_id,
			discharge_coverage=discharge_coverage,
			cell_state_dict=cell_state_dict
		)
		dataset[basin_id] = basin_data_array

	# Do CV on different basins
	r2, mae, coef = fit_probe(
		dataset=dataset
	)
	
	# NVM i need to apply the probe to all basins individually.

	try: 
		out_path = Path("/home/wuhlmann/BA/data/processed_data/probing") / f"{cell_states_path.stem}_domain_probe.p"

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