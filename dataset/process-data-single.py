from DataSyncer import Data

data = Data(
    id="RX0",
    sensor_log_path="dataset/data/chaos-bells-2/raw/RX0-data.log",
    num_sensors=2,
)

data.saveSyncedData("dataset/data/chaos-bells-2/processed/RX0")



