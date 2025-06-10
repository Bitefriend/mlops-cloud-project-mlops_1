import sys
sys.path.append('/home/ubuntu/all_mlops_project/mlops_project')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import torch
import pandas as pd
import boto3
from dotenv import load_dotenv
from datetime import date
import pytz
import glob
from botocore.exceptions import NoCredentialsError

from modeling.src.inference.inference import (
    get_outputs,
    get_scaler_temperature,
    get_scaler_PM,
    inference
)
from modeling.src.model.lstm import MultiOutputLSTM

# .env 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../serving/.env"))

# S3 정보
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL_BUCKET_NAME = "mlops-study-web-lmw"
DATA_BUCKET_NAME = "mlops-pipeline-jeb"
REGION = "ap-northeast-2"

# 경로 설정
DATA_PATH = "/home/ubuntu/all_mlops_project/mlops_data"
MODEL_SAVE_PATH = "/home/ubuntu/all_mlops_project/mlops_data/models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# S3에서 모델 다운로드 함수
def download_model_from_s3():
    print(f"[📦] Connecting to S3 MODEL bucket: {MODEL_BUCKET_NAME}")
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION
    )
    s3.download_file(MODEL_BUCKET_NAME, "models/model_PM_v1.pth", f"{MODEL_SAVE_PATH}/lstm_PM.pth")
    s3.download_file(MODEL_BUCKET_NAME, "models/model_Temperature_v1.pth", f"{MODEL_SAVE_PATH}/lstm_temperature.pth")
    print("✅ 모델 다운로드 완료")

# S3에서 입력 데이터 다운로드 함수
def download_input_data_from_s3():
    def download_data(data_name):
        kst = pytz.timezone('Asia/Seoul')
        now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        ymd = datetime.now(kst).strftime('%Y%m%d')
        prefix = f'results/{data_name}/'
        data_download_path = os.path.join(DATA_PATH, f"s3data/{now}", DATA_BUCKET_NAME)
        os.makedirs(data_download_path, exist_ok=True)

        print(f"[📦] Connecting to S3 DATA bucket: {DATA_BUCKET_NAME}, prefix: {prefix}")

        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
        merged_df_list = []
        continuation_token = None

        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=DATA_BUCKET_NAME, Prefix=prefix, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=DATA_BUCKET_NAME, Prefix=prefix)
            if 'Contents' not in response:
                break

            count = 0
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.csv'):
                    local_path = os.path.join(data_download_path, key)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    try:
                        s3.download_file(DATA_BUCKET_NAME, key, local_path)
                        print(f"✅ Downloaded: s3://{DATA_BUCKET_NAME}/{key} -> {local_path}")
                        df = pd.read_csv(local_path)
                        merged_df_list.append(df)
                        count += 1
                    except Exception as e:
                        print(f"❌ Error reading {key}: {e}")
                if count >= 250:
                    break

            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break

        if merged_df_list:
            new_df = pd.concat(merged_df_list, ignore_index=True)
            output_name = "TA_data.csv" if data_name == "temperature" else "PM10_data.csv"
            new_df.to_csv(os.path.join(DATA_PATH, output_name), index=False)
            print(f"✅ {output_name} 저장 완료")
        else:
            print(f"⚠️ No data downloaded for {data_name}")

    download_data('temperature')
    download_data('pm10')

# 추론 수행 함수
def run_inference():
    print(f"[🔧] 시작: run_inference()")
    print(f"[📁] DATA_PATH = {DATA_PATH}")
    print(f"[📁] MODEL_SAVE_PATH = {MODEL_SAVE_PATH}")
    print(f"[📄] 현재 데이터 폴더 내 파일들: {os.listdir(DATA_PATH)}")

    DEVICE = torch.device("cpu")
    outputs_temperature, outputs_PM = get_outputs()

    df_ta = pd.read_csv(os.path.join(DATA_PATH, "TA_data.csv"))
    df_pm = pd.read_csv(os.path.join(DATA_PATH, "PM10_data.csv"))

    scaler_temp = get_scaler_temperature(DATA_PATH, outputs_temperature)
    scaler_pm = get_scaler_PM(DATA_PATH, outputs_PM)

    model_temp = MultiOutputLSTM(outputs_temperature)
    model_pm = MultiOutputLSTM(outputs_PM)

    model_temp.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "lstm_temperature.pth"), map_location=DEVICE))
    model_pm.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "lstm_PM.pth"), map_location=DEVICE))

    model_temp.to(DEVICE)
    model_pm.to(DEVICE)

    result_temp = inference(model_temp, df_ta[outputs_temperature].values, scaler_temp, outputs_temperature, DEVICE)
    result_pm = inference(model_pm, df_pm[outputs_PM].values, scaler_pm, outputs_PM, DEVICE)

    
    kst = pytz.timezone('Asia/Seoul')
    today = date.today().strftime("%Y-%m-%d")
    pd.DataFrame([result_temp]).to_csv(os.path.join(DATA_PATH, f"result_temp_{today}.csv"), index=False)
    pd.DataFrame([result_pm]).to_csv(os.path.join(DATA_PATH, f"result_pm_{today}.csv"), index=False)

    print("✅ 예측 완료 및 CSV 저장 완료")

# DAG 정의
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    dag_id='daily_inference_dag',
    default_args=default_args,
    schedule_interval='0 20 * * *',  # 매일 살림 4시
    catchup=False,
    tags=['mlops', 'inference', 's3'],
)

t1 = PythonOperator(
    task_id='download_model',
    python_callable=download_model_from_s3,
    dag=dag,
)

t2 = PythonOperator(
    task_id='download_input_data',
    python_callable=download_input_data_from_s3,
    dag=dag,
)

t3 = PythonOperator(
    task_id='run_inference',
    python_callable=run_inference,
    dag=dag,
)

t1 >> t2 >> t3
