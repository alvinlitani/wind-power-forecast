import pandas as pd
from wind_forecast import storage

# adjust paths as needed
mapping = pd.read_csv("data/mapping.csv")
map_names = set(mapping["IESO name"])

# load one IESO file and pull its WIND Generator set
ieso = storage.read_csv("data/processed/ieso/PUB_GenOutputCapabilityMonth_202606.csv")
ieso_names = set(ieso[ieso["Fuel Type"] == "WIND"]["Generator"])

print("in mapping, not IESO:", map_names - ieso_names)
print("in IESO, not mapping:", ieso_names - map_names)