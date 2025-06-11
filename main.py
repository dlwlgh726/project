import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📊 기준금리와 아파트 매매 가격 분석")

# 파일 경로
housing_path = "아파트_매매_실거래_평균가격_20250609090955.csv"
interest_path = "한국은행 기준금리 및 여수신금리_05123930.csv"

# 데이터 불러오기
try:
    housing_raw = pd.read_csv(housing_path, encoding='euc-kr')
    interest_raw = pd.read_csv(interest_path, encoding='utf-8-sig')
except FileNotFoundError:
    st.error("❌ CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    st.stop()

### 1. 아파트 매매 데이터 전처리
try:
    housing_df = housing_raw.copy()
    housing_df = housing_df.rename(columns={housing_df.columns[0]: '지역'})
    housing_df = housing_df[housing_df['지역'] == '전국']
    housing_df = housing_df.drop(columns=['지역'])

    # 연도-월 컬럼명을 '연도'만 추출하여 행으로 변환
    housing_df = housing_df.T.reset_index()
    housing_df.columns = ['연월', '전국']
    housing_df = housing_df[housing_df['연월'].str.match(r'^\d{4}\.\d{2}$')]
    housing_df['연도'] = housing_df['연월'].str.split('.').str[0].astype(int)

    # 연도별 평균값
    housing_df = housing_df.groupby('연도')['전국'].mean().reset_index()
    housing_df['전국'] = housing_df['전국'].astype(int)

except Exception as e:
    st.error(f"❌ 아파트 매매 데이터 처리 중 오류 발생: {e}")
    st.stop()

### 2. 기준금리 데이터 전처리
try:
    interest_df = interest_raw.copy()
    # 5행부터 실제 데이터 시작됨 (E열부터 연도 있음)
    interest_df = interest_df.iloc[4:, 4:]
    interest_df.columns = interest_df.iloc[0]  # 연도 행
    interest_df = interest_df[1:]  # 데이터만 남김
    interest_df = interest_df.T.reset_index()
    interest_df.columns = ['연도', '기준금리']
    interest_df['연도'] = interest_df['연도'].astype(int)
    interest_df['기준금리'] = pd.to_numeric(interest_df['기준금리'], errors='coerce')
    interest_df = interest_df.dropna()
except Exception as e:
    st.error(f"❌ 금리 데이터 처리 중 오류 발생: {e}")
    st.stop()

### 3. 병합
merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')

if merged_df.empty:
    st.warning("⚠️ 병합된 데이터가 없습니다. 연도 범위를 다시 확인하세요.")
    st.stop()

### 4. 시각화
st.subheader("📉 기준금리 vs 전국 아파트 평균 매매가격")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='기준금리', y='전국', ax=ax)
plt.title('기준금리 vs 전국 아파트 평균 매매가격 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
st.pyplot(fig)

### 5. 상관계수
correlation = merged_df['기준금리'].corr(merged_df['전국'])
st.markdown(f"📌 **기준금리와 전국 평균 매매가격의 상관계수**: `{correlation:.3f}`")

### 6. 선형 회귀
X = merged_df[['기준금리']]
y = merged_df['전국']

if len(X) > 0:
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.markdown(f"📈 **회귀식**: `y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}`")
    st.markdown(f"📊 **결정계수 (R²)**: `{r2:.3f}`")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"}, ax=ax2)
    plt.title('기준금리와 전국 평균 매매가격 선형 회귀 분석')
    plt.xlabel('기준금리 (%)')
    plt.ylabel('전국 평균 매매가격 (원)')
    st.pyplot(fig2)
else:
    st.warning("⚠️ 회귀 분석을 위한 데이터가 충분하지 않습니다.")

