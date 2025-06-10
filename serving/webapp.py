from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import pandas as pd
from datetime import datetime

# ✅ 1. 경로 정의
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_PATH = os.path.abspath(os.path.join( "/home/ubuntu/mlops-lmw/mlops-cloud-project-mlops_1/data/results"))

# ✅ 2. FastAPI 앱 초기화 및 CORS 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 3. 예측 결과 조회용 HTML 페이지 (React용으로 꾸며짐)
@app.get("/", response_class=HTMLResponse)
def index():
    today = datetime.now().strftime("%Y년 %m월 %d일 (%A)")
    return f"""
    <!DOCTYPE html>
    <html lang='ko'>
    <head>
        <meta charset='UTF-8'>
        <title>오늘의 날씨 예측 결과</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 2rem; text-align: center; }}
            h1 {{ color: #2d3436; margin-bottom: 0.2rem; }}
            h2 {{ color: #636e72; margin-bottom: 1.5rem; }}
            button {{
                padding: 0.8rem 1.5rem;
                font-size: 1rem;
                background-color: #0984e3;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                margin: 0.5rem;
            }}
            button:hover {{
                background-color: #74b9ff;
            }}
            .result-box {{
                background: #ffffff;
                padding: 2rem;
                margin: 2rem auto;
                width: 100%;
                max-width: 600px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                text-align: left;
                line-height: 1.8;
                font-size: 1rem;
                color: #2d3436;
            }}
        </style>
    </head>
    <body>
        <h1>오늘의 날씨 예측 결과</h1>
        <h2>{today}</h2>
        <button onclick=\"getPrediction()\">예측 요청하기</button>
        <button onclick=\"window.location.reload()\">새로고침</button>
        <div class=\"result-box\" id=\"result\">(예측 결과 없음)</div>
        <script>
            async function getPrediction() {{
                try {{
                    const res = await fetch('/result');
                    if (!res.ok) throw new Error("API 호출 실패");
                    const data = await res.json();
                    function round2(x) {{ return Number.parseFloat(x).toFixed(2); }}
                    let html = `<strong>🌡️ 온도 예측</strong><br>
                        평균 기온: ${{round2(data.temperature_prediction.TA_AVG)}}℃<br>
                        최고 기온: ${{round2(data.temperature_prediction.TA_MAX)}}℃<br>
                        최저 기온: ${{round2(data.temperature_prediction.TA_MIN)}}℃<br><br>
                        <strong>🌫️ 미세먼지 예측</strong><br>
                        평균: ${{round2(data.pm_prediction.PM10_AVG)}} ㎍/㎥<br>
                        최고: ${{round2(data.pm_prediction.PM10_MAX)}} ㎍/㎥<br>
                        최저: ${{round2(data.pm_prediction.PM10_MIN)}} ㎍/㎥`;
                    document.getElementById("result").innerHTML = html;
                }} catch (err) {{
                    document.getElementById("result").textContent = '[에러] ' + '오늘의 날씨 예측 결과가 존재하지 않습니다. 매일 새벽 5시 30분에 업데이트 됩니다.';
                }}
            }}
        </script>
    </body>
    </html>
    """

# ✅ 4. JSON API
@app.get("/result", response_class=JSONResponse)
def get_result():
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        temp_path = os.path.join(DATA_ROOT_PATH, f"result_temp_{today_str}.csv")
        pm_path = os.path.join(DATA_ROOT_PATH, f"result_pm_{today_str}.csv")

        if not os.path.exists(temp_path) or not os.path.exists(pm_path):
            return {
               "error": "오늘의 날씨 예측 결과가 존재하지 않습니다. 매일 새벽 5시 30분에 업데이트 됩니다."
            }

        result_temp = pd.read_csv(temp_path).to_dict(orient="records")[0]
        result_pm = pd.read_csv(pm_path).to_dict(orient="records")[0]

        return {
            "temperature_prediction": result_temp,
            "pm_prediction": result_pm
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
