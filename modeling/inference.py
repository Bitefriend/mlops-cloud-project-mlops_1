import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 30

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

class MultiOutputLSTM(nn.Module):
    def __init__(self, outputs, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(outputs))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
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

def get_engine(db_name):
    engine = create_engine(url=(
        f"mysql+mysqldb://"
        f"{os.environ.get('DB_USER')}:"
        f"{os.environ.get('DB_PASSWORD')}@"
        f"{os.environ.get('DB_HOST')}:"
        f"{os.environ.get('DB_PORT')}/"
        f"{db_name}"))
    return engine

def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)

    connect = engine.connect()
    data.to_sql(table_name, connect, if_exists="append")
    connect.close()

def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    connect = engine.connect()
    result = connect.execute(
        statement=text(
            f"select recommend_content_id from {table_name} "
            f"order by `index` desc limit :k"
        ),
        parameters={"table_name": table_name, "k": k}
    )
    connect.close()
    contents = [data[0] for data in result]
    return contents


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
    
    run_inference()

