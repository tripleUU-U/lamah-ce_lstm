import sys
import pickle
import time 
import logging

from pathlib import Path
from sklearn.manifold import MDS

def mds_basin(cell_states_path: Path, basin_id: str):

    with open(cell_states_path, "rb") as f:
        cell_state_dict = pickle.load(f)
        cell_states = cell_state_dict[basin_id]["c_last"]

    logging.info(f"Starting MDS for Basin {basin_id}, Shape: {cell_states.shape}")

    out_dict = {}
    
    for n in [2, 3]: 

        mds = MDS(
            n_components=n,
            n_init=4,
            metric=True,
            random_state=1277,
            n_jobs=-1,
            max_iter=1000,
            verbose=2,
            normalized_stress=True
        )

        mds.fit_transform(cell_states)

        logger.info(f"Final stress {n}D: {mds.stress_}")

        out_dict[n] = mds

    try:
        out_path = cell_states_path.parent / f"{cell_states_path.stem}_{basin_id}_mds.p"
        
        with open(out_path, "wb") as out:
            pickle.dump(out_dict, out)

        logging.info(f"Gespeichert unter: {out_path}")

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