import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 파일 불러오기
housing_df = pd.read_csv('아파트_매매_실거래_평균가격_20250609090955.csv', encoding='euc-kr', thousands=',')
interest_df = pd.read_csv('한국은행 기준금리 및 여수신금리_05123930.csv', encoding='utf-8-sig', thousands=',')

# 2. 날짜 컬럼명 정리
housing_df.rename(columns={housing_df.columns[0]: '날짜'}, inplace=True)
interest_df.rename(columns={interest_df.columns[0]: '날짜'}, inplace=True)

# 3. 날짜 형식 변환 (에러 나는 값은 NaT로 변환)
housing_df['날짜'] = pd.to_datetime(housing_df['날짜'], format='%Y.%m', errors='coerce')
interest_df['날짜'] = pd.to_datetime(interest_df['날짜'], errors='coerce')

# 4. 변환 실패(NaT) 값 제거
housing_df.dropna(subset=['날짜'], inplace=True)
interest_df.dropna(subset=['날짜'], inplace=True)

# 5. 아파트 데이터에서 연도만 추출 (int 타입)
housing_df['연도'] = housing_df['날짜'].dt.year

# 6. 금리 데이터에서 연도만 추출 (int 타입)
interest_df['연도'] = interest_df['날짜'].dt.year

# 7. 필요한 열만 선택
housing_df = housing_df[['연도', '전국']]
interest_df = interest_df[['연도', '기준금리']]

# 8. 2006년 이후 데이터만 필터링 (아파트 기준)
housing_df = housing_df[housing_df['연도'] >= 2006]
interest_df = interest_df[interest_df['연도'] >= 2006]

# 9. 연도 기준 병합 (inner join)
merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')

# 10. 결측치 제거
merged_df.dropna(inplace=True)

# 11. 시각화
plt.figure(figsize=(10,6))
sns.scatterplot(data=merged_df, x='기준금리', y='전국')
plt.title('기준금리 vs 전국 아파트 평균 매매가격 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()

# 12. 상관계수
correlation = merged_df['기준금리'].corr(merged_df['전국'])
print(f"📌 기준금리와 전국 평균 매매가격의 상관계수: {correlation:.3f}")

# 13. 단순 선형 회귀
X = merged_df[['기준금리']]
y = merged_df['전국']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\n📈 회귀식: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
print(f"📊 결정계수 R²: {r2:.3f}")

# 14. 회귀선 시각화
plt.figure(figsize=(10,6))
sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"})
plt.title('기준금리와 전국 평균 매매가격 선형 회귀 분석 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()
