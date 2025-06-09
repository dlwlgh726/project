import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 아파트 매매 데이터 불러오기 (헤더 없이 불러온 후 직접 처리)
housing_raw = pd.read_csv('아파트_매매_실거래_평균가격_20250609090955.csv', encoding='euc-kr', header=None, thousands=',')

# 2. '전국' 데이터는 2번째 행(B2부터)
housing_dates = housing_raw.iloc[0, 1:]  # 1행, 2열부터 날짜 (예: '2006.01')
housing_nation = housing_raw.iloc[1, 1:]  # 2행, 2열부터 전국 데이터

# 3. 데이터프레임 생성
housing_df = pd.DataFrame({
    '연월': housing_dates,
    '전국': housing_nation
})

# 4. '연월'에서 연도만 추출해 정수형으로 변환
housing_df['연도'] = housing_df['연월'].str.split('.').str[0].astype(int)

# 5. 2006년 이후 데이터만 필터링
housing_df = housing_df[housing_df['연도'] >= 2006].copy()

# 6. 금리 데이터 불러오기 (헤더 없이 불러오기)
interest_raw = pd.read_csv('한국은행 기준금리 및 여수신금리_05123930.csv', encoding='utf-8-sig', header=None, thousands=',')

# 7. 금리 데이터는 E열(4번째 인덱스)부터 연도 및 값 있음
interest_years = interest_raw.iloc[0, 4:].astype(str)  # 첫 행, 5열부터 연도
interest_rates = interest_raw.iloc[1, 4:].astype(float)  # 두번째 행, 5열부터 금리 값

# 8. 금리 데이터프레임 생성 및 연도 필터링
interest_df = pd.DataFrame({
    '연도': interest_years.astype(int),
    '기준금리': interest_rates
})
interest_df = interest_df[interest_df['연도'] >= 2006].copy()

# 9. 필요한 열만 선택 (housing_df는 이미 연도, 전국 포함)
housing_df = housing_df[['연도', '전국']]

# 10. 두 데이터 연도 기준 병합
merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')

# 11. 결측치 제거
merged_df.dropna(inplace=True)

# 12. 시각화
plt.figure(figsize=(10,6))
sns.scatterplot(data=merged_df, x='기준금리', y='전국')
plt.title('기준금리 vs 전국 아파트 평균 매매가격 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()

# 13. 상관계수 계산
correlation = merged_df['기준금리'].corr(merged_df['전국'])
print(f"📌 기준금리와 전국 평균 매매가격의 상관계수: {correlation:.3f}")

# 14. 단순 선형 회귀
X = merged_df[['기준금리']]
y = merged_df['전국']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\n📈 회귀식: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
print(f"📊 결정계수 R²: {r2:.3f}")

# 15. 회귀선 시각화
plt.figure(figsize=(10,6))
sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"})
plt.title('기준금리와 전국 평균 매매가격 선형 회귀 분석 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()
