# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="금리와 아파트 가격 예측", layout="wide")
st.title("🏡 금리와 아파트 매매가격의 관계 분석 (2006~2024)")

# 파일 업로드
apt_file = st.file_uploader("📁 아파트 매매 실거래 평균가격 CSV 업로드", type="csv")
rate_file = st.file_uploader("📁 한국은행 금리 CSV 업로드", type="csv")

if apt_file and rate_file:
    try:
        # CSV 불러오기
        apt_df = pd.read_csv(apt_file, encoding="cp949")
        rate_df = pd.read_csv(rate_file, encoding="cp949")

        # 날짜 컬럼 정리
        apt_df = apt_df.rename(columns={apt_df.columns[0]: "날짜"})
        rate_df = rate_df.rename(columns={rate_df.columns[0]: "날짜"})

        apt_df["날짜"] = pd.to_datetime(apt_df["날짜"], errors="coerce")
        rate_df["날짜"] = pd.to_datetime(rate_df["날짜"], errors="coerce")

        apt_df = apt_df.dropna(subset=["날짜"])
        rate_df = rate_df.dropna(subset=["날짜"])

        # 연도 추출
        apt_df["연도"] = apt_df["날짜"].dt.year
        rate_df["연도"] = rate_df["날짜"].dt.year

        # 연도 필터링: 2006~2024년
        apt_df = apt_df[(apt_df["연도"] >= 2006) & (apt_df["연도"] <= 2024)]
        rate_df = rate_df[(rate_df["연도"] >= 2006) & (rate_df["연도"] <= 2024)]

        # 연도별 평균 계산
        apt_year = apt_df.groupby("연도").mean(numeric_only=True).reset_index()
        rate_year = rate_df.groupby("연도").mean(numeric_only=True).reset_index()

        # 병합
        merged = pd.merge(apt_year, rate_year, on="연도", how="inner")

        if merged.empty:
            st.error("🚫 병합된 데이터가 없습니다. 연도가 겹치지 않거나 데이터에 문제가 있을 수 있습니다.")
            st.stop()

        # 주요 컬럼 자동 선택
        price_col = [col for col in merged.columns if "가격" in col][0]
        rate_col = [col for col in merged.columns if "금리" in col][0]

        # 시각화
        st.subheader("📈 연도별 금리 vs 아파트 평균가격")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(merged["연도"], merged[price_col], color='tab:blue', marker='o', label='아파트 가격')
        ax1.set_ylabel("아파트 가격", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(merged["연도"], merged[rate_col], color='tab:red', marker='s', label='기준금리')
        ax2.set_ylabel("기준금리", color="tab:red")
        st.pyplot(fig)

        # 회귀 모델 학습
        st.subheader("🔍 선형 회귀 분석")
        X = merged[[rate_col]]
        y = merged[price_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"**📊 R² Score:** `{r2_score(y_test, y_pred):.4f}`")
        st.markdown(f"**📈 회귀 계수 (기울기):** `{model.coef_[0]:.2f}`")
        st.markdown(f"**📉 절편:** `{model.intercept_:.2f}`")

        # 회귀 시각화
        fig2, ax = plt.subplots()
        sns.regplot(x=rate_col, y=price_col, data=merged, ax=ax)
        ax.set_xlabel("기준금리")
        ax.set_ylabel("아파트 가격")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
else:
    st.info("⏳ 두 개의 CSV 파일을 업로드해 주세요.")
