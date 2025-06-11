# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="금리와 아파트 가격", layout="wide")
st.title("📊 금리와 아파트 매매가격 상관관계 및 예측 모델")

# 파일 업로드
apt_file = st.file_uploader("📁 아파트 매매 실거래 평균가격 파일을 업로드하세요 (CSV)", type="csv")
rate_file = st.file_uploader("📁 한국은행 기준금리 및 여수신금리 파일을 업로드하세요 (CSV)", type="csv")

if apt_file and rate_file:
    try:
        apt_df = pd.read_csv(apt_file, encoding="cp949")
        rate_df = pd.read_csv(rate_file, encoding="cp949")

        # 날짜 처리
        apt_df = apt_df.rename(columns={apt_df.columns[0]: "날짜"})
        rate_df = rate_df.rename(columns={rate_df.columns[0]: "날짜"})
        apt_df["날짜"] = pd.to_datetime(apt_df["날짜"], errors="coerce")
        rate_df["날짜"] = pd.to_datetime(rate_df["날짜"], errors="coerce")

        apt_df = apt_df.dropna(subset=["날짜"])
        rate_df = rate_df.dropna(subset=["날짜"])

        # 월 단위로 평균 내기
        apt_df = apt_df.set_index("날짜").resample("M").mean().reset_index()
        rate_df = rate_df.set_index("날짜").resample("M").mean().reset_index()

        # 병합
        merged = pd.merge(apt_df, rate_df, on="날짜", how="inner")

        # 자동 컬럼 인식
        price_col = [col for col in merged.columns if "가격" in col][0]
        rate_col = [col for col in merged.columns if "금리" in col][0]

        # 시각화
        st.subheader("📈 금리와 아파트 매매가 시계열")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(merged["날짜"], merged[price_col], color='tab:blue', label='아파트 가격')
        ax1.set_ylabel("아파트 평균가격", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(merged["날짜"], merged[rate_col], color='tab:red', label='기준금리')
        ax2.set_ylabel("기준금리", color="tab:red")
        st.pyplot(fig)

        # 회귀 분석
        st.subheader("🤖 단순 선형 회귀 예측")
        X = merged[[rate_col]]
        y = merged[price_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"**R² Score (설명력):** `{r2_score(y_test, y_pred):.4f}`")
        st.markdown(f"**회귀 계수 (기울기):** `{model.coef_[0]:.2f}`")
        st.markdown(f"**절편:** `{model.intercept_:.2f}`")

        # 산점도 + 회귀선
        fig2, ax = plt.subplots()
        sns.regplot(x=rate_col, y=price_col, data=merged, ax=ax)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
else:
    st.info("⏳ 두 개의 CSV 파일을 업로드해 주세요.")
