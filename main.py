import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Streamlit 제목
st.title("📊 기준금리와 전국 아파트 평균 매매가격 분석")

# 파일 업로드
housing_file = st.file_uploader("아파트 매매 실거래 평균가격 파일 업로드", type=['csv'])
interest_file = st.file_uploader("한국은행 기준금리 파일 업로드", type=['csv'])

# 분석 시작 조건
if housing_file is not None and interest_file is not None:
    try:
        # 1. 아파트 가격 데이터 로딩
        housing_raw = pd.read_csv(housing_file, encoding='euc-kr', thousands=',')
        housing_raw = housing_raw.rename(columns={housing_raw.columns[0]: '지역'})
        housing_df = housing_raw[housing_raw['지역'] == '전국'].drop('지역', axis=1)
        housing_df = housing_df.T.reset_index()
        housing_df.columns = ['연월', '전국']
        housing_df = housing_df[housing_df['연월'].str.match(r'^\d{4}\.\d{2}$')]
        housing_df['연도'] = housing_df['연월'].str[:4].astype(int)
        housing_df = housing_df[['연도', '전국']]
        housing_df['전국'] = housing_df['전국'].astype(float)
        housing_df = housing_df.groupby('연도').mean().reset_index()
        housing_df = housing_df[housing_df['연도'] >= 2006]

        # 2. 기준금리 데이터 로딩
        interest_raw = pd.read_csv(interest_file, encoding='utf-8-sig')
        interest_df = interest_raw.iloc[1:, :]
        interest_df.columns = interest_raw.iloc[0]
        interest_df = interest_df.dropna(axis=1, how='all')

        year_columns = [str(y) for y in range(2006, 2026) if str(y) in interest_df.columns]
        interest_df = interest_df[year_columns]
        interest_df = interest_df.astype(float)

        interest_df = pd.DataFrame({
            '연도': [int(col) for col in interest_df.columns],
            '기준금리': interest_df.mean(axis=0).values
        })

        # 3. 데이터 병합
        merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')
        merged_df.dropna(inplace=True)

        # 4. 시각화: 산점도
        st.subheader("🔍 기준금리와 전국 평균 매매가격 산점도")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='기준금리', y='전국', ax=ax1)
        ax1.set_title('기준금리 vs 전국 아파트 평균 매매가격 (연도별)')
        ax1.set_xlabel('기준금리 (%)')
        ax1.set_ylabel('전국 평균 매매가격 (원)')
        ax1.grid(True)
        st.pyplot(fig1)

        # 5. 상관계수
        correlation = merged_df['기준금리'].corr(merged_df['전국'])
        st.write(f"📌 **기준금리와 전국 평균 매매가격의 상관계수**: `{correlation:.3f}`")

        # 6. 단순 선형 회귀 분석
        X = merged_df[['기준금리']]
        y = merged_df['전국']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        st.write(f"📈 **회귀식**: `y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}`")
        st.write(f"📊 **결정계수 R²**: `{r2:.3f}`")

        # 7. 회귀선 시각화
        st.subheader("📈 선형 회귀 분석 시각화")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"}, ax=ax2)
        ax2.set_title('기준금리와 전국 평균 매매가격 선형 회귀 분석 (연도별)')
        ax2.set_xlabel('기준금리 (%)')
        ax2.set_ylabel('전국 평균 매매가격 (원)')
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")
else:
    st.info("📁 두 CSV 파일을 모두 업로드해 주세요.")
