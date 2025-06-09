import pandas as pd

# 파일 경로 설정
housing_file = '아파트_매매_실거래_평균가격_20250609090955.csv'
interest_file = '한국은행 기준금리 및 여수신금리_05123930.csv'

# 인코딩 적용해서 열 이름만 확인
housing_df = pd.read_csv(housing_file, encoding='euc-kr', nrows=1)
interest_df = pd.read_csv(interest_file, encoding='utf-8-sig', nrows=1)

print("🏠 아파트 매매 데이터 열 이름:")
print(housing_df.columns.tolist())

print("\n🏦 금리 데이터 열 이름:")
print(interest_df.columns.tolist())
