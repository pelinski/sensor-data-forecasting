# sensor-data-forecasting
This code loads a dataset collected using [bela-data-logger](https://github.com/pelinski/bela-data-logger) and [bela-data-syncer](https://github.com/pelinski/bela-data-syncer) and trains a neural network for next sample prediction. For now, it is only possible to train an lstm. 


## Usage
You can create an environment with the necessary dependencies using pipenv:
```
pipenv install
```
You will also need to install torch:
```
pipenv run pip3 install torch
```
You can sync data coming from multiple Belas using the script in `data/process-data-multi.py`. If you have data coming only from one Bela, you can process it using `process-data-single.py`.

Once the dataset is processed, you can modify the necessary paths in `train.py` and run the training script by typing:
```
pipenv run python train.py
```
The training and model parameters can be modified by passing `.yaml` files to the wandb config:
```
pipenv run python train.py --config configs/test-trans.yaml
```