# Nasdaq Linear Regression Forecasting Project

이 프로젝트는 2024년 나스닥 종합지수(IXIC)의 일봉 종가를 선형회귀 모델로 예측하는 것을 목표로 합니다.

## 개요

- **목표**: 2024년 나스닥 종합지수(IXIC) 일봉 종가를 선형회귀로 예측
- **데이터**: FRED API를 통한 시계열 데이터 자동 수집
- **특징**:
    - 기술적 파생 피처(지수 Lag, SMA) 및 거시지표 조합
    - 다중공선성(VIF)과 상관관계(Pearson)를 기반한 변수 선택
    - StandardScaler와 선형회귀 파이프라인
    - 시계열 기반 Train-Test 분할(70% / 30%)

## 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
# Python 3.9 이상 필요
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 필요 라이브러리 설치

```bash
# 필수 라이브러리 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
.
├── plb1_ixic_linear.py       # 메인 스크립트
├── data/                     # 데이터 저장 폴더
│   ├── NASDAQCOM.csv         # 나스닥 지수 데이터