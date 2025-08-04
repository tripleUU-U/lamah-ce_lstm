import re
import pandas as pd 

from pathlib import Path
from scipy.interpolate import make_splrep 
from cuml.tsa import SARIMA
from concurrent.futures import ThreadPoolExecutor

def spline_impute():

    pass

def sarima_impute():

    pass

def impute_timeseries(
    discharge_timeseries: pd.DataFrame,
    coverage_data: pd.DataFrame
) -> pd.Series: 

    discharge_timeseries["date"] = pd.to_datetime(
        discharge_timeseries[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
    )
    discharge_timeseries.set_index("date", inplace=True)

    # Copy the original discharge,
    discharge_timeseries["qimp"] = discharge_timeseries["qobs"]

    for gap in discharge_timeseries["gap_dates"]: 

        start_date, end_date = gap

        gap_length = end_date - start_date

        if gap_length <= 2: 

            spline_impute
        
        else:
            sarima_impute



def main():

    discharge_data_path = Path(r"/home/wuhlmann/BA/data/raw_data/2_LamaH-CE_daily/D_gauges/2_timeseries/daily")
    timeseries_paths = list(sorted(discharge_data_path.glob("*csv")))

    coverage_data_path = Path(r"/home/wuhlmann/BA/data/processed_data/eda/discharge_coverage.csv")

    coverage_df = pd.read_table(coverage_data_path, header=0, sep=";")

    target_basin_ids = list(coverage_df[coverage_df["num_missing_days"] != 0]["basin_id"])

    for path in timeseries_paths: 
        
        basin_id = int(re.search(r"\d+", path.stem).group())

        # Only impute basins with missing data. 
        if not basin_id in target_basin_ids: 
            continue

        discharge_timeseries = pd.read_table(path, header=0, sep=";")

        impute_timeseries(
            discharge_timeseries=discharge_timeseries,
            coverage_data=coverage_df
        )

        

# load coverage, get ids of basins with missing q, load q, get gap dates, go over coverage, if gap if small gap, spline, if laerger train sarima, l