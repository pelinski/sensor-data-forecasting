from DataSyncer import DataSyncerTX, DataSyncerRX

dataSyncerTX = DataSyncerTX(
    id="TX0",
    sync_log_path="data/raw/TX0-sync.log",
    sensor_log_path="data/raw/TX0-data.log",
    num_sensors=8,
    d_clock=689 * 8 + 8,
)

dataSyncerTX.saveSyncedData("data/synced/TX0")

dataSyncerRX1 = DataSyncerRX(id="RX1",
                             sync_log_path="data/raw/RX1-sync1.log",
                             sensor_log_path="data/raw/RX1-data1.log",
                             num_sensors=8)
dataSyncerRX2 = DataSyncerRX(id="RX2",
                             sync_log_path="data/raw/RX2-sync1.log",
                             sensor_log_path="data/raw/RX2-data1.log",
                             num_sensors=8)

dataSyncerRX1.syncSensorData(dataSyncerTX)
dataSyncerRX1.saveSyncedData("data/synced/RX1")
dataSyncerRX2.syncSensorData(dataSyncerTX)
dataSyncerRX2.saveSyncedData("data/synced/RX2")
