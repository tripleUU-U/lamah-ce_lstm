import sys
import pickle
import time 
import logging

import numpy as np

from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

def mds_domain(
    cell_states_path: Path
): 
       
    with open(cell_states_path, "rb") as f:
       cell_state_dict = pickle.load(f)

    out_dict = defaultdict(dict)

    # Sort so that the states match the id order of basins.
    cell_state_dict = dict(sorted(cell_state_dict.items(), key=lambda item: int(item[0])))

    # Extract the mean state of every basin.
    cell_states = np.vstack([cell_state_dict[b]["c_mean"] for b in cell_state_dict.keys()])

    for dim in [2,3]:
    
        mds= MDS(
                n_components=dim,
                n_init=2,
                random_state=1277,
                n_jobs=-1
            )

        logging.info(f"Calculating MDS for doman with shape {cell_states.shape} using: \n {mds.get_params()}.")

        out_dict[str(dim)]["raw"] = mds.fit_transform(cell_states)
        logging.info(f"Finished MDS with raw values.")

        out_dict[str(dim)]["normal"] = mds.fit_transform(StandardScaler().fit_transform(cell_states))
        logging.info(f"Finished MDS with normalized values.")

     # Pickle results.
    try: 
        out_path = cell_states_path.parent / f"{cell_states_path.stem}_means_mds.p"

        with open(out_path, "wb") as out: 
            pickle.dump(out_dict, out)

        logger.info(f"Cell states successfully saved at: {out_path}")

    except Exception as e: 

        logger.error(f"Pickling of cell states failed:\n{e}")

def main():

    cell_states_path = Path(sys.argv[1])
    
    mds_domain(
        cell_states_path=cell_states_path
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