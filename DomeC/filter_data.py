import numpy as np

from DomeC.process_dome_c_data import prepare_dome_c_data

# Load data and bring it into correct format
data = prepare_dome_c_data()

# Fill missing time points
data = data.dropna(subset=["timestamp"]).set_index("timestamp").asfreq("10Min")

# Label data which is not in u bin with nan
data.loc[(data["U2[m s-1]"] < (5.5 - 1.5)) | (data["U2[m s-1]"] > (5.5 + 1.5))] = np.nan

# Remove all columns which are not relevant
data.drop(data.columns.difference(["U2[m s-1]"]), 1, inplace=True)

# Detrend data (i.e. remove influence from scales bigger than submeso)
# 1. Calculate rolling mean
data["MA"] = data["U2[m s-1]"].rolling(window=3).mean()
#  2. Calculate difference
data["diff"] = data["U2[m s-1]"] - data["MA"]

# Split data back up into continuous time series and get variance for every subsequence
# 1. Mark rows which belong to same continuous time series
data["cumsum_nan"] = data["diff"].isna().cumsum()
# 2. Calculate variance for all subsequences
data = data.groupby(["cumsum_nan"]).var()
# 3. Remove rows with nan
data.dropna(axis="rows", inplace=True)

# Final result: mean of all variances from the continuous time series which are in the u bin
print(np.mean(data["diff"]))
