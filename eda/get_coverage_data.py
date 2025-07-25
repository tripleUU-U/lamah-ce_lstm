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
    valid_basin_ids = list(static_attributes["ID"])

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

        for day in discharge_timeseries.index:

            if discharge_timeseries.loc[day, "qobs"] == -999: 

                # If there is no current gap but a NA, start a new one. 
                if current_gap_len == 0: 

                    current_gap_date_range[0] = day

                current_gap_len += 1

                # If there is gap on the last day of the timeseries, set current date as end date of the gap.  
                if day == discharge_timeseries.index[ts_len-1]:

                    # Reset not necessary, since its the last day. 
                    current_gap_date_range[1] = day
                    gap_date_ranges.append(current_gap_date_range)
                    
            else: 

                # If there was a gap, its now over, append gap length and reset counter. 
                if current_gap_len > 0:
                    gap_lengths.append(current_gap_len)
                    current_gap_len = 0

                    # Also add the prior date, as the last gap day.
                    current_gap_date_range[1] = day - pd.Timedelta(days=1)

                    gap_date_ranges.append(current_gap_date_range)
                    current_gap_date_range = [None, None]
        
        if gap_lengths: 
            average_gap_lengths.append(np.mean(gap_lengths))
            max_gap_lengths.append(np.max(gap_lengths))
            gap_dates.append(gap_date_ranges)

        else: 
            average_gap_lengths.append(0)
            max_gap_lengths.append(0)
            gap_dates.append(None)

        # Create basin states df to display in logger. 
        basin_stats = pd.DataFrame([{
            "basin_id": basin_id,
            "timeseries_length": ts_len,
            "start_date": start_dates[-1],
            "end_date": end_dates[-1],
            "num_missing_days": count_NA,
            "percentage [%]": gap_percentages[-1],
            "average_gap_length": average_gap_lengths[-1],
            "max_gap_length": max_gap_lengths[-1],
            "gap_dates": gap_dates[-1]
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
        "gap_dates": gap_dates
    })

    # Sort after basin id and set as index.
    coverage_stats_sorted = coverage_stats.sort_values("basin_id")
    coverage_stats_sorted.set_index("basin_id", inplace=True)

    # Write to disk.
    processed_data_path = Path("/home/wuhlmann/BA/data/processed_data")
    coverage_stats_sorted.to_csv(
        processed_data_path / "eda/discharge_coverage.csv",
        sep=";"
    )

if __name__ == "__main__": 

    logging.basicConfig(
        level=logging.INFO,              
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    main()




