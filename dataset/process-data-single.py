from DataSyncer import Data

data = Data(
    id="RX0",
    sensor_log_path="dataset/data/chaos-bells-2/raw/RX0-data.log",
    num_sensors=2,
)

data.saveSyncedData("dataset/data/chaos-bells-2/processed/RX0")

## test data (sliced version of dataset)
# f = open("dataset/data/test-data/RX0", 'w+b')
# binary_format = bytearray(data.sensor_np[0:1000])
# f.write(binary_format)
# f.close()

