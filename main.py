import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 파일 불러오기
housing_path = "아파트_매매_실거래_평균가격_20250609090955 (1).csv"
interest_path = "한국은행 기준금리 및 여수신금리_05123930 (1).csv"

housing_df = pd.read_csv(housing_path, encoding='euc-kr')
interest_df = pd.read_csv(interest_path, encoding='utf-8-sig')

# 2. 아파트 데이터 전처리
# - 행정구역 이름을 인덱스로 설정
housing_df = housing_df.set_index(housing_df.columns[0])
# - 행(전국) 선택 후 전치
housing_df = housing_df.loc['전국'].T.reset_index()
housing_df.columns = ['연도월', '전국']
# - 연도 추출
housing_df['연도'] = housing_df['연도월'].str.split('.').str[0].astype(int)
# - 연도별 평균값 계산
housing_df = housing_df.groupby('연도')['전국'].mean().reset_index()

# 3. 금리 데이터 전처리
# - 필요한 행(0행)에서 데이터 추출
interest_df = interest_df.iloc[0]
# - 필요한 열만 선택 (2001~2025 사이 연도만)
interest_df = interest_df[['2006','2007','2008','2009','2010','2011','2012',
                           '2013','2014','2015','2016','2017','2018','2019',
                           '2020','2021','2022','2023','2024','2025']]
# - Series -> DataFrame
interest_df = interest_df.T.reset_index()
interest_df.columns = ['연도', '기준금리']
interest_df['연도'] = interest_df['연도'].astype(int)
interest_df['기준금리'] = interest_df['기준금리'].astype(float)

# 4. 데이터 병합
merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')
merged_df.dropna(inplace=True)

# 5. 시각화 - 산점도
plt.figure(figsize=(10,6))
sns.scatterplot(data=merged_df, x='기준금리', y='전국')
plt.title('기준금리 vs 전국 아파트 평균 매매가격 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()

# 6. 상관계수 계산
correlation = merged_df['기준금리'].corr(merged_df['전국'])
print(f"📌 기준금리와 전국 평균 매매가격의 상관계수: {correlation:.3f}")

# 7. 단순 선형 회귀 분석
X = merged_df[['기준금리']]
y = merged_df['전국']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\n📈 회귀식: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
print(f"📊 결정계수 R²: {r2:.3f}")

# 8. 회귀선 시각화
plt.figure(figsize=(10,6))
sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"})
plt.title('기준금리와 전국 평균 매매가격 선형 회귀 분석 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()
