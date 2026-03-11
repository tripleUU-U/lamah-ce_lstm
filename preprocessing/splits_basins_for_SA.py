from pathlib import Path
import numpy as np

# Path to the .txt holding the training basin ids.
basin_id_path = Path("/home/wuhlmann/BA/repo/lamah-ce_lstm/data/splits_incl_landcover/train_basin_ids.txt") 
base_out_path = Path("/home/wuhlmann/BA/data/processed_data/SA/train_basin_splits")

with open(basin_id_path, "r") as f:
    # Skip last one, to avoid searching for basin " ".
    ids_list = f.read().split("\n")[:-1]

splits = np.array_split(ids_list, 7)

for splits in splits: 

    out_path = base_out_path / f"basin{splits[0]}-{splits[-1]}_ids.txt"

    with open(out_path, "w") as out: 
        for id in splits:
            out.write(f"{id}\n")
