import re
import logging

import pandas as pd
import numpy as np

from pathlib import Path 

def main():

    # Define path objects to data.
    data_directory_path = Path("/home/wuhlmann/BA/data")
    discharge_data_path = data_directory_path / "raw_data/2_LamaH-CE_daily/D_gauges/2_timeseries/daily"

    # Static attribute file is needed to filter out basins, without static catchment attributes.
    static_attributes_path = data_directory_path / "raw_data/2_LamaH-CE_daily/B_basins_intermediate_all/1_attributes/Catchment_attributes.csv"
    static_attributes = pd.read_table(static_attributes_path, header=0, sep=";")
    static_attributes.set_index("ID", inplace=True)
    valid_basin_ids = list(static_attributes.index)

    # Get gauge attributes for real area. 
    gauge_attributes_path = data_directory_path / "raw_data/2_LamaH-CE_daily/D_gauges/1_attributes/Gauge_attributes.csv"
    gauge_attributes = pd.read_table(gauge_attributes_path, header=0, sep=";")
    gauge_attributes.set_index("ID", inplace=True)

    # Get Hydrological indices. 
    hydro_indices_path = data_directory_path / "raw_data/2_LamaH-CE_daily/D_gauges/1_attributes/Hydro_indices_1981_2017.csv"
    hydro_indices = pd.read_table(hydro_indices_path, header=0, sep=";")
    hydro_indices.set_index("ID", inplace=True)

    # Gather path objects of all discharge timeseries. 
    discharge_timeseries_paths = list(sorted(discharge_data_path.glob("*.csv")))

    # Coverage data. 
    basin_ids = []
    timeseries_lengths = []
    start_dates = []
    end_dates = []
    
    # Normal periods.
    n_normal_days = []
    q5_q95_normal_periods = []

    for path in discharge_timeseries_paths:
        
        # Extract basin id from filename. 
        basin_id = int(re.search(r"\d+", path.stem).group())

        if not basin_id in valid_basin_ids: 
            # Skip invalid files. 
            continue

        basin_ids.append(basin_id)

        # Read discharge timeseries. 
        discharge_timeseries = pd.read_table(path, header=0, sep=";")

        ts_len = len(discharge_timeseries)
        timeseries_lengths.append(ts_len)

        # Add timestamp index. 
        discharge_timeseries["date"] = pd.to_datetime(
            discharge_timeseries[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
        )
        discharge_timeseries.set_index("date", inplace=True)

        start_dates.append(discharge_timeseries.index[0])
        end_dates.append(discharge_timeseries.index[ts_len - 1])

        # Prepare tracking of periods between Q5 and Q95. 
        q5 = hydro_indices.loc[basin_id, "Q5"] 
        q95 = hydro_indices.loc[basin_id, "Q95"]

        # Q5 and Q95 are averaged over the area, revert.
        area = gauge_attributes.loc[basin_id, "area_gov"]
        q5 = q5 / (1000 * 86400) * area * 1e6
        q95 = q95 / (1000 * 86400) * area * 1e6

        calc_q5 = np.percentile(discharge_timeseries["qobs"],5)
        calc_q95 = np.percentile(discharge_timeseries["qobs"],95)

        current_normal_period_duration = 0 
        current_normal_period_date_span = [None, None]

        basin_n_normal_days = 0
        basin_q_normal_periods = []

        for day in discharge_timeseries.index:

            qobs_day = discharge_timeseries.loc[day, "qobs"]

            # Track "normal" periods with q5 < qobs < q95. 
            if qobs_day > q5 and qobs_day < q95: 

                # Normal period starts. 
                if current_normal_period_duration == 0: 
                    current_normal_period_date_span[0] = day

                current_normal_period_duration += 1
            else: 
                # If there was a normal period, its now over. 
                if current_normal_period_duration > 0: 
                    current_normal_period_date_span[1] = day - pd.Timedelta(days=1)
                    
                    if current_normal_period_duration >= 0:
                        basin_q_normal_periods.append(current_normal_period_date_span)
                
                # Add number of normal days of this period to total. 
                basin_n_normal_days += current_normal_period_duration

                current_normal_period_duration = 0 
                current_normal_period_date_span = [None, None]

        if basin_q_normal_periods:
            n_normal_days.append(basin_n_normal_days)
            q5_q95_normal_periods.append(basin_q_normal_periods)
        else:
            n_normal_days.append(0)
            q5_q95_normal_periods.append([])

        # Create basin stats df to display in logger. 
        basin_stats = pd.DataFrame([{
            "basin_id": basin_id,
            "timeseries_length": ts_len,
            "start_date": start_dates[-1],
            "end_date": end_dates[-1],
            "n_normal_days": n_normal_days[-1],
            "q5_q95": q5_q95_normal_periods[-1]
        }])
        logger.info("\n%s", basin_stats)


    # Construct final coverage dataframe. 
    coverage_stats = pd.DataFrame({
        "basin_id": basin_ids,
        "timeseries_length": timeseries_lengths,
        "start_date": start_dates,
        "end_date" : end_dates,
        "n_normal_days": n_normal_days,
        "q5_q95_normal_periods": q5_q95_normal_periods,
    })

    # Sort after basin id and set as index.
    coverage_stats_sorted = coverage_stats.sort_values("basin_id")
    coverage_stats_sorted.set_index("basin_id", inplace=True)


    # Write to disk.
    processed_data_path = Path("/home/wuhlmann/BA/data/processed_data")
    coverage_stats_sorted.to_pickle(
        processed_data_path / "eda/normal_periods.pkl"
    )

if __name__ == "__main__": 

    logging.basicConfig(
        level=logging.INFO,              
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    main()
