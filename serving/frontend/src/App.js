import React, { useState } from 'react';

function App() {
  const [result, setResult] = useState('');

  const getPrediction = async () => {
    try {
      const res = await fetch("/result");  // 프록시를 통해 백엔드로 전달
      if (!res.ok) throw new Error("API 호출 실패");
      const data = await res.json();
      setResult(JSON.stringify(data, null, 2));
    } catch (err) {
      setResult('[에러] ' + "오늘의 날씨 예측 결과가 존재하지 않습니다. 매일 새벽 5시 30분에 업데이트 됩니다.");
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial' }}>
      <h1>오늘의 예측 결과 (React)</h1>
      <button onClick={getPrediction}>예측 요청하기</button>
      <pre>{result || '(예측 결과 없음)'}</pre>
    </div>
  );
}

export default App;