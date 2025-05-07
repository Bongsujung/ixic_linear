
"""
plb1_ixic_linear.py
───────────────────────────────────────────────────────────────────────────────
• 목적  : 2024년 나스닥 종합지수(IXIC) 일봉 종가를 선형회귀로 예측
• 특징  : * FRED API 자동 수집 및 CSV 캐싱
          - 기술적 파생(지수 Lag·SMA) + 거시지표 피처 구성
          - 다중공선성(VIF)·상관(Pearson) 기반 변수 선택
          - StandardScaler → LinearRegression 파이프라인
          - Train–Test 시계열 분할(70 % / 30 %)
          - 모델 계수·R² 출력 + 전처리 CSV 저장
───────────────────────────────────────────────────────────────────────────────
요구 라이브러리 : Python ≥ 3.9, pandas ≥ 2, scikit-learn ≥ 1,
                requests, statsmodels (VIF 계산용, 선택 사항)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 0.  표준 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import io
import re
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# 1.  서드파티 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
import requests                     # HTTP GET (FRED API)
import pandas as pd                 # 데이터프레임/시계열 처리
import sklearn.preprocessing as pp # 스케일링 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# (옵션) VIF 계산용. statsmodels 가 없다면 주석 처리해도 실행은 됨
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _HAS_STATSMODELS = True
except ImportError:                 # statsmodels 미설치 환경 대응
    _HAS_STATSMODELS = False

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SSL 인증서 경고 무시
#     ‣ macOS 의 루트 체인 문제로 HTTPS verify 에러가 잦아, warning 만 off
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  헬퍼 함수 : FRED 시계열 다운로드
# ─────────────────────────────────────────────────────────────────────────────
def fred_series(series_id: str) -> pd.Series:
    """
    FRED(세인트루이스 연준) 데이터베이스에서 특정 시리즈를
    • 우선 `.txt`(legacy) 엔드포인트로 시도
    • HTML 이면 `.csv` 엔드포인트로 폴백
    • DatetimeIndex + float 값의 pandas Series 로 반환

    Parameters
    ----------
    series_id : str
        FRED Series ID, 예) 'NASDAQCOM', 'SP500', ...

    Returns
    -------
    pd.Series
        이름(name) 이 series_id 로 지정된 시계열

    Notes
    -----
    - 결측치는 '.' 으로 표시됨 → float 변환 시 건너뜀
    - 함수 내부에서 /data/{series_id}.csv 로 매번 캐싱
    """
    # 3-1) .txt (legacy) endpoint -------------------------------------------
    txt_url = f"https://fred.stlouisfed.org/graph/fredgraph.txt?id={series_id}"
    r_txt = requests.get(txt_url, timeout=30, verify=False)
    raw_txt = r_txt.text.lstrip()

    # 3-2) 응답이 HTML → CSV 엔드포인트로 재시도 -----------------------------
    if raw_txt.startswith("<"):
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df = pd.read_csv(csv_url, index_col=0, parse_dates=True)
        except Exception:  # macOS + SSL 문제 우회
            r_csv = requests.get(csv_url, timeout=30, verify=False)
            df = pd.read_csv(io.StringIO(r_csv.text), index_col=0, parse_dates=True)

        # 헤더명이 예측 불가할 때: 첫 번째 컬럼을 사용
        s = df[series_id] if series_id in df.columns else df.iloc[:, 0]
    else:
        # 3-3) .txt 파싱 ------------------------------------------------------
        rows: list[tuple[str, float]] = []
        for ln in raw_txt.splitlines():
            if ln.startswith(("DATE", "#")) or not ln.strip():
                continue                       # 헤더·주석·빈줄 skip
            date_str, val_str = ln.split()
            if val_str == ".":                 # 누락치
                continue
            rows.append((date_str, float(val_str)))

        idx = pd.to_datetime([d for d, _ in rows], format="%Y-%m-%d")
        s = pd.Series([v for _, v in rows], index=idx, name=series_id)

    # 3-4) 로컬 캐싱 (data/{ID}.csv) -----------------------------------------
    data_dir = Path(__file__).with_suffix("").parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{series_id}.csv").write_text(
        s.to_csv(header=["value"]), encoding="utf-8"
    )

    return s.astype(float).rename(series_id)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  데이터 수집 (2024-01-02 ~ 2024-12-31)
#     * 첫 거래일 1/2, 마지막 12/31 – 휴장일은 자동 결측
# ─────────────────────────────────────────────────────────────────────────────
target = fred_series("NASDAQCOM").loc["2024"]   # 종속변수
sp500  = fred_series("SP500").loc["2024"]        # 미국 S&P500
djia   = fred_series("DJIA").loc["2024"]         # 다우 30
vix    = fred_series("VIXCLS").loc["2024"]       # 변동성지수 (후보)
dgs10  = fred_series("DGS10").loc["2024"]        # 10년물 국채금리 (후보)
wti    = fred_series("DCOILWTICO").loc["2024"]   # 서부텍사스유
btc    = fred_series("CBBTCUSD").loc["2024"]     # 비트코인 USD

# ─────────────────────────────────────────────────────────────────────────────
# 5.  기술적 파생 피처 (Lag·SMA)
# ─────────────────────────────────────────────────────────────────────────────
lag1  = target.shift(1).rename("IXIC_LAG1")          # 전일 종가
lag5 = target.shift(5).rename("IXIC_LAG5")          # 5일 전 종가
sma5  = target.rolling(5).mean().rename("IXIC_SMA5") # 5일 이동평균
sma10 = target.rolling(10).mean().rename("IXIC_SMA10") #10일 이동평균

# ─────────────────────────────────────────────────────────────────────────────
# 6.  피처 풀 (10개) 결합 → 결측 제거(dropna)
# ─────────────────────────────────────────────────────────────────────────────
feat_full = pd.concat(
    [lag1, lag5, sma5, sma10, sp500, djia, vix, dgs10, wti, btc],
    axis=1
)
data = pd.concat([target.rename("IXIC"), feat_full], axis=1).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  상관계수(절댓값) 순으로 정렬
# ─────────────────────────────────────────────────────────────────────────────
corr = (
    data.corr(numeric_only=True)       # 일부 pandas 버전에서 경고 방지
        .loc["IXIC"]
        .abs()
        .sort_values(ascending=False)
)

# 7-1) 출력 및 확인 (필요 시)
print("\n[상관계수 상위 10]\n", corr.head(10), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  후보 → 최종 4개 피처 선택
#     ─ Pearson r ≥ 0.4 & VIF < 10 기준
# ─────────────────────────────────────────────────────────────────────────────
top4_cols = ["IXIC_LAG1", "IXIC_SMA5", "SP500", "DJIA"]  # 수작업 선택

# 8-1) (선택) VIF 출력
if _HAS_STATSMODELS:
    vif = pd.Series(
        [variance_inflation_factor(data[top4_cols].values, i)
         for i in range(len(top4_cols))],
        index=top4_cols, name="VIF"
    )
    print("[VIF]\n", vif, "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  학습/테스트 분할 (시계열 ⇒ shuffle=False)
# ─────────────────────────────────────────────────────────────────────────────
X = data[top4_cols]
y = data["IXIC"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=False # 시계열 분할, test_size=30%
)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  스케일링 + 선형회귀 모델 학습
# ─────────────────────────────────────────────────────────────────────────────
scaler = pp.StandardScaler().fit(X_train) # 학습 데이터 평균 및 표준편차 계산
# 10-1) 스케일링 (표준화)
#     - StandardScaler() : 평균 0, 표준편차 1
#     - MinMaxScaler()   : 0 ~ 1 범위로 변환
#     - RobustScaler()   : 중앙값 0, IQR 1
#     - MaxAbsScaler()   : 절대값 최대 1
#     - Normalizer()     : L2 정규화
#     - QuantileTransformer() : 분위수 변환
#     - PowerTransformer() : Box-Cox 변환
#     - PolynomialFeatures() : 다항식 변환


X_train_std = scaler.transform(X_train) # 학습 데이터 변환
X_test_std  = scaler.transform(X_test)# 테스트 데이터 변환

# 10-2) 다양한 선형회귀 모델 생성

'''
기본 선형회구 모델 : LinearRegression

대안 모델들 : Ridge, Lasso, ElasticNet
'''
# 기본 선형회귀 모델
#model = LinearRegression()  # 선형회귀 모델

# 대안 모델들 (필요시 주석 해제하여 사용)
# model = Ridge(alpha=1.0)  # L2 정규화 (과적합 방지)
# model = Lasso(alpha=0.1)  # L1 정규화 (변수 선택 효과)
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # L1+L2 혼합 정규화, 이놈이 최고 성능  # Train R² : 0.9717, Test  R² : 0.9135

model.fit(X_train_std, y_train) # 모델 학습
print(f"학습 데이터 크기 : {X_train_std.shape}")
print(f"테스트 데이터 크기 : {X_test_std.shape}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  예측 및 평가
# ─────────────────────────────────────────────────────────────────────────────
y_pred_tr = model.predict(X_train_std) # 학습 데이터 예측
y_pred_te = model.predict(X_test_std) # 테스트 데이터 예측

r2_tr = r2_score(y_train, y_pred_tr) # 학습 데이터 R²
r2_te = r2_score(y_test,  y_pred_te) # 테스트 데이터 R²

print(f"Train R² : {r2_tr:.4f}")
print(f"Test  R² : {r2_te:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  회귀 계수(표준화 이후 계수) 출력
# ─────────────────────────────────────────────────────────────────────────────
coef_ser = pd.Series(model.coef_, index=top4_cols, name="β") # 회귀 계수
print("회귀계수\n--------")
print(coef_ser, "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  전처리된 통합 데이터 저장 (재현성 확보)
# ─────────────────────────────────────────────────────────────────────────────
preproc_dir = Path(__file__).with_suffix("").parent / "data" / "preproc"
preproc_dir.mkdir(parents=True, exist_ok=True)
data.to_csv(preproc_dir / "ixic_2024_preprocessed.csv")

print(f"전처리 데이터 저장 완료 → {preproc_dir/'ixic_2024_preprocessed.csv'}")