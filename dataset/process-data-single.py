from DataSyncer import Data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", help="dataset path", type=str)
parser.add_argument("-s","--save_path", help="save path",type=str)
parser.add_argument("-i","--id", help="Bela ID",type=int)
parser.add_argument("-n","--num_sensors", help="number of sensors",type=int)
args = parser.parse_args()

data = Data(
    id=args.id,
    sensor_log_path=args.dataset,
    num_sensors=args.num_sensors,
)

data.saveSyncedData(args.save_path)


