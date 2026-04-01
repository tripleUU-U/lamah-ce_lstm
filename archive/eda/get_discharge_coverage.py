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

    # Missing days data. 
    num_missing_days = []    
    gap_percentages = []
    average_gap_lengths = []
    max_gap_lengths = []
    gap_dates = []

    # Normal periods.
    q5_q95_normal_periods = []
    z_normal_periods = []

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

        # Get total missings days. 
        count_NA = (discharge_timeseries["qobs"] == -999).sum()
        num_missing_days.append(count_NA)

        gap_percentages.append(count_NA/ts_len*100)

        # Get data on individual gaps. 
        gap_lengths = []
        gap_date_ranges = []

        # Counter & temporary holder. 
        current_gap_len = 0
        current_gap_date_range = [None, None]

        # Prepare tracking of periods between Q5 and Q95. 
        q5 = hydro_indices.loc[basin_id, "Q5"] 
        q95 = hydro_indices.loc[basin_id, "Q95"]

        # Q5 and Q95 are averaged over the area, revert.
        area = static_attributes.loc[basin_id, "area_calc"]
        q5 = (q5 * area * 10e6) / (1000* 86400)
        q95 = (q95 * area * 10e6) / (1000 * 86400)
        
        current_normal_period_duration = 0 
        current_normal_period_date_span = [None, None]

        basin_q_normal_periods = []
        
        # Calculate mean and std for z-score. 
        q_mean = discharge_timeseries["qobs"].mean()
        q_std = discharge_timeseries["qobs"].std()

        current_z_duration = 0 
        current_z_date_span = [None, None]

        basin_z_normal_periods = []

        for day in discharge_timeseries.index:

            qobs_day = discharge_timeseries.loc[day, "qobs"]

            if qobs_day == -999: 

                # If there is no current gap but a NA, start a new one. 
                if current_gap_len == 0: 

                    current_gap_date_range[0] = str(day)

                current_gap_len += 1

                # If there is gap on the last day of the timeseries, set current date as end date of the gap.  
                if day == discharge_timeseries.index[ts_len-1]:

                    # Reset not necessary, since its the last day. 
                    current_gap_date_range[1] = str(day)
                    gap_date_ranges.append(current_gap_date_range)
                    
            else: 

                # If there was a gap, its now over, append gap length and reset counter. 
                if current_gap_len > 0:
                    gap_lengths.append(current_gap_len)
                    current_gap_len = 0

                    # Also add the prior date, as the last gap day.
                    current_gap_date_range[1] = str(day - pd.Timedelta(days=1))

                    gap_date_ranges.append(current_gap_date_range)
                    current_gap_date_range = [None, None]
        
            z_score = (qobs_day - q_mean) / q_std

            # Track z score normal periods. 
            if abs(z_score) < 2.0: 

                # Z Normal period starts. 
                if current_z_duration == 0: 
                    current_z_date_span[0] = day

                current_z_duration += 1

            else: 

                # If there was a normal period, its now over. 
                if current_z_duration > 0: 
                    current_z_date_span[1] = day - pd.Timedelta(days=1)
                    
                    # Append normal period to list.
                    basin_z_normal_periods.append(current_z_date_span) 
                
                current_z_duration = 0 
                current_z_date_span = [None, None]

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
                
                current_normal_period_duration = 0 
                current_normal_period_date_span = [None, None]

        if gap_lengths: 
            average_gap_lengths.append(np.mean(gap_lengths))
            max_gap_lengths.append(np.max(gap_lengths))
            gap_dates.append(gap_date_ranges)

        else: 
            average_gap_lengths.append(0)
            max_gap_lengths.append(0)
            gap_dates.append([])

        if basin_z_normal_periods:
            z_normal_periods.append(basin_z_normal_periods)
        else:
            z_normal_periods.append([])

        if basin_q_normal_periods:
            q5_q95_normal_periods.append(basin_z_normal_periods)
        else:
            q5_q95_normal_periods.append([])

        # Create basin stats df to display in logger. 
        basin_stats = pd.DataFrame([{
            "basin_id": basin_id,
            "timeseries_length": ts_len,
            "start_date": start_dates[-1],
            "end_date": end_dates[-1],
            "num_missing_days": count_NA,
            "percentage [%]": gap_percentages[-1],
            "average_gap_length": average_gap_lengths[-1],
            "max_gap_length": max_gap_lengths[-1],
            "gap_dates": gap_dates[-1],
            "z_normal_period": z_normal_periods[-1],
            "q5_q95": q5_q95_normal_periods[-1]
        }])
        logger.info("\n%s", basin_stats)


    # Construct final coverage dataframe. 
    coverage_stats = pd.DataFrame({
        "basin_id": basin_ids,
        "timeseries_length": timeseries_lengths,
        "start_date": start_dates,
        "end_date" : end_dates,
        "num_missing_days": num_missing_days,
        "percentage [%]": gap_percentages,
        "average_gap_length": average_gap_lengths,
        "max_gap_length": max_gap_lengths,
        "gap_dates": gap_dates,
        "q5_q95_normal_periods": q5_q95_normal_periods,
        "z_normal_periods": z_normal_periods 
    })

    # Sort after basin id and set as index.
    coverage_stats_sorted = coverage_stats.sort_values("basin_id")
    coverage_stats_sorted.set_index("basin_id", inplace=True)


    # Write to disk.
    processed_data_path = Path("/home/wuhlmann/BA/data/processed_data")
    coverage_stats_sorted.to_pickle(
        processed_data_path / "eda/discharge_coverage.pkl"
    )

if __name__ == "__main__": 

    logging.basicConfig(
        level=logging.INFO,              
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    main()
