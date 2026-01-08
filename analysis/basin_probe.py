# utils
import sys
import time
import logging
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

#sk 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

def probe_basin(
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

	model = ElasticNet()
	kfold = KFold(n_splits=5)
	folds = list(kfold.split(cell_states, temp.values))

	best_r2 = -100
	best_model_id = None
	model_list = []

	# Do 5 fold validation and let the best model predict (probe) on the entire timeseries.
	for i, f in zip([0, 1, 2, 3, 4], folds):
		
		train_ids = f[0]
		test_ids = f[1]

		target = temp.iloc[test_ids]

		model.fit(X=cell_states[train_ids], y=temp.iloc[train_ids])
		pred = model.predict(X=cell_states[test_ids])

		# Collect all models.
		model_list.append(model)

		r2 = r2_score(target, pred)

		if r2 > best_r2:
			best_r2 = r2
			best_model_id = i

	best_model = model_list[best_model_id]
	
	total_pred = best_model.predict(cell_states)
	total_r2 = r2_score(temp, total_pred)
	total_mae = mean_absolute_error(temp, total_pred)

	# Only the coeffiecents of the model are needed, since it's not used again. 
	coef_list = best_model.coef_.tolist()

	return total_r2, total_mae, coef_list


def main():

	cell_states_path = Path(sys.argv[1])
	target_var = str(sys.argv[2])

	logger.info(f"Setting up basin level probes for {target_var} on {cell_states_path.stem} cell states.")

	with open(cell_states_path, "rb") as f: 
		cell_state_dict = pickle.load(f)
		
	cell_state_dict = dict(sorted(cell_state_dict.items(), key=lambda item: int(item[0])))

	# Get discharge coverage, to slice timeseries according to discharge timeseries.
	with open("/home/wuhlmann/BA/data/processed_data/eda/normal_periods.pkl", "rb") as f: 
		discharge_coverage = pickle.load(f)

	ts_path = Path("/home/wuhlmann/BA/data/raw_data/2_LamaH-CE_daily/B_basins_intermediate_all/2_timeseries/daily")

	# Collect r2 and model coefficients for each basement. 
	probe_results_dict = defaultdict(dict)

	for basin_id in tqdm(cell_state_dict.keys()):

		r2, mae, coef = probe_basin(
			ts_path=ts_path,
			target_var=target_var,
			basin_id=int(basin_id),
			discharge_coverage=discharge_coverage,
			cell_state_dict=cell_state_dict
		)

		logger.info(f"Basin {basin_id} probe reached R2 of {r2:.2f} and MAE of {mae:.2f}.")

		probe_results_dict[basin_id]["R2"] = r2
		probe_results_dict[basin_id]["MAE"] = mae
		probe_results_dict[basin_id]["coef"] = coef


	try: 
		out_path = Path("/home/wuhlmann/BA/data/processed_data/probing") / f"{cell_states_path.stem}_{target_var}_basin_probe.p"

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