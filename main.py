import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 파일 경로 지정
housing_path = "/mnt/data/아파트_매매_실거래_평균가격_20250609090955.csv"
interest_path = "/mnt/data/한국은행 기준금리 및 여수신금리_05123930.csv"

# 2. 아파트 매매 데이터 전처리
housing_raw = pd.read_csv(housing_path, encoding='euc-kr')

# 컬럼명은 첫 번째 행, 데이터는 두 번째 행 사용
header = housing_raw.iloc[0, 1:]   # '2006.01', '2006.02', ...
data = housing_raw.iloc[1, 1:]     # 평균 가격

# 새로운 DataFrame 생성
housing_df = pd.DataFrame({
    '연월': header.values,
    '전국': data.values
})

# 연도 추출
housing_df['연도'] = housing_df['연월'].astype(str).str.split('.').str[0].astype(int)
housing_df['전국'] = pd.to_numeric(housing_df['전국'], errors='coerce')  # 문자열 -> 숫자

# 연도별 평균 계산 (같은 연도 여러 월 존재 → 연도별 평균)
housing_df = housing_df.groupby('연도', as_index=False)['전국'].mean()

# 2006년 이후만 필터링
housing_df = housing_df[housing_df['연도'] >= 2006]

# 3. 금리 데이터 전처리
interest_raw = pd.read_csv(interest_path, encoding='utf-8-sig')

# '기준금리'가 포함된 행만 필터링 (2번째 열 기준)
rate_row = interest_raw[interest_raw.iloc[:, 1].astype(str).str.contains('기준금리', na=False)]

# 금리 데이터 추출 (E열 이후 연도 데이터)
rate_values = rate_row.iloc[:, 4:].transpose()
rate_values.columns = ['기준금리']
rate_values.index.name = '연도'
rate_values.reset_index(inplace=True)

# 연도 int 변환 및 금리 float 변환
rate_values['연도'] = rate_values['연도'].astype(int)
rate_values['기준금리'] = pd.to_numeric(rate_values['기준금리'], errors='coerce')

# 2006년 이후 필터링
interest_df = rate_values[rate_values['연도'] >= 2006]

# 4. 병합
merged_df = pd.merge(housing_df, interest_df, on='연도', how='inner')
merged_df.dropna(inplace=True)

# 5. 상관관계 시각화
plt.figure(figsize=(10, 6))
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
plt.figure(figsize=(10, 6))
sns.regplot(x='기준금리', y='전국', data=merged_df, ci=None, line_kws={"color": "red"})
plt.title('기준금리와 전국 평균 매매가격 선형 회귀 분석 (연도별)')
plt.xlabel('기준금리 (%)')
plt.ylabel('전국 평균 매매가격 (원)')
plt.grid(True)
plt.show()
