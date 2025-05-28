import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


from dotenv import load_dotenv

import pandas as pd
import numpy as np
import torch

from modeling.src.inference.inference import (
    init_model, inference, temperature_to_df, write_db, PM_to_df, get_scalers, get_outputs
)
from modeling.src.train.train import (
    train
)

WINDOW_SIZE = 30

def run_train_temperature(outputs_temperature, scaler_temperature):
    data = pd.read_csv('../mlops_data/TA_data.csv')

    train(data, outputs_temperature, scaler_temperature, "Temperature")

def run_train_PM(outputs_PM, scaler_PM):
    data = pd.read_csv('../mlops_data/PM10_data.csv')

    train(data, outputs_PM, scaler_PM, "PM")

def run_train():

    scaler_temperature, scaler_PM = get_scalers()
    outputs_temperature, outputs_PM = get_outputs()

    run_train_temperature(outputs_temperature, scaler_temperature)
    run_train_PM(outputs_PM, scaler_PM)

def run_inference_temperature(model, scaler, outputs, device):
    fake_test_data = np.random.normal(loc=15, scale=3, size=(WINDOW_SIZE, len(outputs)))

    results = inference(model, fake_test_data, scaler, outputs, device)    
    temperature_df = temperature_to_df(results, outputs)
    print(temperature_df)
    write_db(temperature_df, "mlops", "temperature")

def run_inference_PM(model, scaler, outputs, device):
    fake_test_data = np.random.normal(loc=15, scale=3, size=(WINDOW_SIZE, len(outputs)))

    results = inference(model, fake_test_data, scaler, outputs, device)    
    PM_df = PM_to_df(results, outputs)
    print(PM_df)
    write_db(PM_df, "mlops", "PM")

def run_inference(batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model_temperature, model_PM = init_model()
    scaler_temperature, scaler_PM = get_scalers()
    outputs_temperature, outputs_PM = get_outputs()

    run_inference_temperature(model_temperature, scaler_temperature, outputs_temperature, device)
    run_inference_PM(model_PM, scaler_PM, outputs_PM, device)


if __name__ == '__main__':

    load_dotenv()

    # run_inference()

    run_train()


    