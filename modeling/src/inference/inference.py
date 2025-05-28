import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


from modeling.src.model.lstm import MultiOutputLSTM

def temperature_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

def PM_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

def get_outputs():
    outputs_temperature = ["TA_AVG", "TA_MAX", "TA_MIN"]
    outputs_PM = ["PM10_MIN", "PM10_MAX", "PM10_AVG"]
    return outputs_temperature, outputs_PM

def get_scaler_temperature(outputs):
    df = pd.read_csv('../mlops_data/TA_data.csv')
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def get_scaler_PM(outputs):
    df = pd.read_csv('../mlops_data/PM10_data.csv')
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def get_scalers():
    outputs_temperature, outputs_PM = get_outputs()
    return get_scaler_temperature(outputs_temperature), get_scaler_PM(outputs_PM)

def init_model():
    outputs_temperature, outputs_PM = get_outputs()
    
    model_temperature_path = "../models/lstm_temperature.pth"

    model_temperature_checkpoint = torch.load(model_temperature_path, map_location=torch.device('cpu'), weights_only=True)

    model_temperature = MultiOutputLSTM(outputs_temperature)
    model_temperature.load_state_dict(model_temperature_checkpoint)

    model_PM_path = "../models/lstm_PM.pth"

    model_PM_checkpoint = torch.load(model_PM_path, map_location=torch.device('cpu'), weights_only=True)

    model_PM = MultiOutputLSTM(outputs_PM)
    model_PM.load_state_dict(model_PM_checkpoint)

    return model_temperature, model_PM

def inference_temperature(model, data, scaler, outputs, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_scaled = scaler.transform(data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        prediction = output.cpu().numpy().squeeze()
        result = scaler.inverse_transform([prediction])
    return {
        outputs[0]: result[0][0], 
        outputs[1]: result[0][1], 
        outputs[2]: result[0][2]}

def inference(model, data, scaler, outputs, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_scaled = scaler.transform(data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        prediction = output.cpu().numpy().squeeze()
        result = scaler.inverse_transform([prediction])
    return {
        outputs[0]: result[0][0], 
        outputs[1]: result[0][1], 
        outputs[2]: result[0][2]}
