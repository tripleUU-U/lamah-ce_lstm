import sys
import pickle
import time 
import logging

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

def mds_basin(
	cell_states_path: Path,
	basin_id: str
): 
	
	mds = MDS(
		n_components=3,
		n_init=1,
		metric=False,
		random_state=1277,
		n_jobs=-1,
		verbose=2
	)

	with open(cell_states_path, "rb") as f:
		cell_state_dict = pickle.load(f)

		# Only take the states of the requested basin.
		cell_states = cell_state_dict[basin_id]["c_last"]

	out_dict = {}

	logging.info(f"Calculating MDS for basin {basin_id} with shape {cell_states.shape} using: \n {mds.get_params()}.")

	out_dict["raw"] = mds.fit_transform(cell_states)
	logging.info(f"Finished MDS with raw values.")

	# Pickle results.
	try: 
		out_path = cell_states_path.parent / f"{cell_states_path.stem}_{basin_id}.p"

		with open(out_path, "wb") as out: 
			pickle.dump(out_dict, out)

		logger.info(f"Cell states successfully saved at: {out_path}")

	except Exception as e: 

		logger.error(f"Pickling of cell states failed:\n{e}")

def main():

	cell_states_path = Path(sys.argv[1])
	basin_id = str(sys.argv[2])

	mds_basin(
		cell_states_path=cell_states_path,
		basin_id=basin_id
	)

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

	logger.info(f"MDS completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")