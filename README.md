# BERT 감성분석 Streamlit PyCharm 프로젝트

BERT Fine-tuning 구조를 `src_jungjaehee/` 코드로 분리하고, Streamlit 앱에서 문장을 입력하면 긍정/부정 분류 결과가 출력되도록 구성했습니다.
영어 리뷰 분석에 더해, 한국어 리뷰 문장을 입력하면 사전 학습된 한국어 감성분석 모델로 결과를 출력하는 기능을 추가했습니다.

## 1. 프로젝트 구조
```
Bert_sentiment_project/
├─ data/
│  └─ sample_sentiment.csv      # 실행 테스트용 작은 예제 데이터
├─ models/
│  └─ .gitkeep                  # 학습 모델 저장 폴더
├─ src_jungjaehee/
│  ├─ app/
│  │  └─ streamlit_app.py       # Streamlit 화면 실행 파일 (영어 + 한국어)
│  ├─ __init__.py
│  ├─ config.py                 # 경로, 모델명, 라벨 설정
│  ├─ data_loader.py            # CSV 로드와 train/valid/test 분리
│  ├─ dataset.py                # BERT 입력용 PyTorch Dataset
│  ├─ modeling.py               # BERT 모델 생성과 Fine-tuning 전략
│  ├─ predict.py                # 영어 문장 예측 클래스
│  ├─ predict_kr.py             # 한국어 문장 예측 클래스 (사전 학습 모델)
│  ├─ train.py                  # 모델 학습 실행 스크립트
│  └─ utils.py                  # 시드 고정, 장치 선택 함수
├─ requirements.txt             # 설치 패키지 목록
├─ .gitignore                   # Git 제외 파일 설정
└─ README.md                    # 실행 방법 설명
```

## 2. PyCharm에서 실행 준비
### 2-1. 프로젝트 만들기
PyCharm에서 `Bert_sentiment_project` 폴더를 생성합니다.

### 2-3. 패키지 설치
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3. 모델 학습 (영어)
기본 예제 데이터로 빠르게 구조를 확인하려면 아래 명령을 실행합니다.
```bash
python -m src_jungjaehee.train --data_path data/sample_sentiment.csv --epochs 1 --train_batch_size 2 --eval_batch_size 2
```
실제 IMDB 데이터셋을 사용할 경우 CSV 파일에 다음 컬럼이 있어야 합니다.
```text
review,sentiment
영화 리뷰 문장 또는 영어 리뷰 문장,positive
영화 리뷰 문장 또는 영어 리뷰 문장,negative
```

실제 데이터 파일을 사용하는 예시는 다음과 같습니다.
```bash
python -m src_jungjaehee.train --data_path "data/IMDB Dataset.csv" --epochs 1 --train_batch_size 8 --eval_batch_size 16
```

학습이 끝나면 모델과 토크나이저가 아래 경로에 저장됩니다.
```text
models/bert_sentiment/
```

한국어 리뷰 분석은 별도 학습 과정 없이, `src_jungjaehee/predict_kr.py`에서 사전 학습된 한국어 감성분석 모델(`sangrimlee/bert-base-multilingual-cased-nsmc`)을 바로 불러와 사용합니다.

## 4. Streamlit 앱 실행
```bash
streamlit run src_jungjaehee/app/streamlit_app.py
```

브라우저 화면에서 영어 또는 한국어 문장을 입력한 뒤 각각의 `감성분석 실행` 버튼을 누르면 다음 결과가 출력됩니다.
- 긍정 또는 부정 분류 결과
- 긍정 확률
- 부정 확률
- 현재 사용한 모델 경로

## 5. 코드 구성 핵심
이 프로젝트는 실제 PyCharm 프로젝트에서 관리하기 쉬운 앱 구조로 분리했습니다.
- `src_jungjaehee/data_loader.py`: CSV 파일을 읽고 라벨을 숫자로 변환합니다.
- `src_jungjaehee/dataset.py`: 문장을 BERT 입력 텐서로 변환합니다.
- `src_jungjaehee/modeling.py`: BERT 분류 모델 생성과 Fine-tuning 전략을 처리합니다.
- `src_jungjaehee/train.py`: 학습, 평가, 저장을 실행합니다.
- `src_jungjaehee/predict.py`: 저장된 영어 BERT 모델을 불러와 문장을 예측합니다.
- `src_jungjaehee/predict_kr.py`: 사전 학습된 한국어 BERT 모델을 불러와 문장을 예측합니다.
- `src_jungjaehee/app/streamlit_app.py`: 사용자가 영어/한국어 문장을 입력하고 결과를 확인하는 화면입니다.

## 6. 주의사항
영어 분석의 기본 모델은 `bert-base-uncased`입니다.
이 모델은 영어 데이터에 적합합니다.
한국어 문장 감성분석은 `src_jungjaehee/predict_kr.py`의 사전 학습된 모델을 사용하며,
직접 학습한 모델로 바꾸고 싶다면 한국어 데이터셋으로 학습한 뒤 해당 파일의 모델명을 교체하면 됩니다.