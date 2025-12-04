# utils
import sys
import logging
import time
import numpy as np
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

# parallel
from joblib import Parallel, delayed

# DL
import torch
from torch.utils.data import DataLoader

# NH
from neuralhydrology.datasetzoo import get_dataset 
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.utils.config import Config

def get_basin_ids_from_txt(
    id_txt_path: Path
) -> list[str]: 

    with open(id_txt_path, "r") as f:
        ids_list = f.read().split("\n")
    
    # Skip last one, to avoid searching for basin " ".
    return ids_list[:-1]

def extract_state_per_basin( 
    run_dir: Path, 
    cfg: Config, 
    phase: str, 
    scaler,
    basin_id: str
) -> tuple[str, dict]: 
    """Parallalizable function, to extract cell state and average it."""

    basin_dict = {
        "c_mean": None,
        "c_last": None
    }

    # Construct cuda:id string. 
    device = torch.device("cuda:0")

    # Initialize model with base paramters from config. 
    ea_lstm = EALSTM(cfg=cfg)

    # Load the trained weights into the model. 
    weights_path = run_dir / "model_epoch030.pt"
    weights = torch.load(str(weights_path), map_location=device)
    ea_lstm.load_state_dict(weights)

    # Set to eval, to deactivate dropout.
    ea_lstm.to(device)
    ea_lstm.eval()

    # load dataset for basin_id 
    ds = get_dataset(cfg=cfg, is_train=False, period=phase, scaler=scaler, basin=basin_id)
 
    # Pass the entire basin at once.
    dataloader = DataLoader(ds, batch_size=15000, shuffle=False, collate_fn=ds.collate_fn, num_workers=0, pin_memory=True)

    with torch.no_grad():
        for data in dataloader: 

            # Move batch to device
            for key in data.keys():
                    if key.startswith('x_d'):
                        data[key] = {k: v.to(device) for k, v in data[key].items()}
                    elif not key.startswith('date'):
                        data[key] = data[key].to(device)

            model_output = ea_lstm(data) 

            # c_n has shape num_samples * 365 * 256
            c_n = model_output["c_n"].cpu().numpy()
 
            # Only take the cell state last day (prediction) for visualisation with shape num_samples * 256
            c_last = c_n[:,-1,:]
            c_last = c_last[~np.isnan(c_last).any(axis=1)]
            basin_dict["c_last"] = c_last 

            # vstack to get a long array with shape (num_samples * 365) * 256 and remove nan from warmup
            c_stack = np.vstack(c_n)
            c_stack = c_stack[~np.isnan(c_stack).any(axis=1)]

            # Calculate average cell state over all states occupied. 
            basin_dict["c_mean"] = c_stack.mean(axis=0)

            # Free memory 
            del c_stack, c_last, c_n, model_output, data
            torch.cuda.empty_cache()

    return basin_id, basin_dict

def main(): 

    run_dir = Path(sys.argv[1])
    cfg = Config(run_dir / "config.yml")
    scaler = load_scaler(run_dir=run_dir)

    cell_states_list = []

    logger.info(f"Initialized script with {run_dir}")
    
    pbar = tqdm(["train","validation", "test"])

    for phase in pbar: 

        pbar.set_description(f"Extracting from {phase} phase.")

        path_phase = phase if phase != "validation" else "val"

        phase_basin_ids = get_basin_ids_from_txt(
            Path(f"/home/wuhlmann/BA/data/processed_data/test_splits/{path_phase}_basin_ids.txt")
        )

        parallel_extractor = Parallel(n_jobs=2, verbose=10)

        phase_results_list = parallel_extractor(
            delayed(extract_state_per_basin)(run_dir=run_dir, cfg=cfg, phase=phase, scaler=scaler, basin_id=basin_id) 
            for basin_id in phase_basin_ids
            )
        
        # Attach the new basins to existing list, to enable easy conversion to dict. 
        cell_states_list.extend(phase_results_list)

    cell_states_dict = dict(cell_states_list)

    # Pickle results.
    try: 

        out_path = Path("/home/wuhlmann/BA/data/processed_data/cell_states") / f"{run_dir.stem}_cell_states.p"

        with open(out_path, "wb") as out: 
            pkl.dump(cell_states_dict, out)

        logger.info(f"Cell states successfully saved at: {out_path}")
        exit_code = True

    except Exception as e: 

        logger.error(f"Pickling of cell states failed:\n{e}")

        exit_code = False

    return exit_code

if __name__ == "__main__": 

    logging.basicConfig(
        level=logging.INFO,              
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()

    exit_code = main()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    
    if exit_code:
        logger.info(f"Cell state extraction completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    else: 
        logger.info(f"Cell state extraction failed and wasted {hours:02d}:{minutes:02d}:{seconds:05.2f}")



