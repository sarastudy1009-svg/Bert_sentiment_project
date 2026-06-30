# BERT 감성분석 Streamlit PyCharm 프로젝트

BERT Fine-tuning 구조를 `src/` 코드로 분리하고, Streamlit 앱에서 문장을 입력하면 긍정/부정 분류 결과가 출력되도록 구성했습니다.

## 1. 프로젝트 구조
```
Bert_sentiment_project/
├─ app/
│  └─ streamlit_app.py          # Streamlit 화면 실행 파일
├─ data/
│  └─ sample_sentiment.csv      # 실행 테스트용 작은 예제 데이터
├─ models/
│  └─ .gitkeep                  # 학습 모델 저장 폴더
├─ src/
│  ├─ config.py                 # 경로, 모델명, 라벨 설정
│  ├─ data_loader.py            # CSV 로드와 train/valid/test 분리
│  ├─ dataset.py                # BERT 입력용 PyTorch Dataset
│  ├─ modeling.py               # BERT 모델 생성과 Fine-tuning 전략
│  ├─ predict.py                # 문장 예측 클래스
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

## 3. 모델 학습
기본 예제 데이터로 빠르게 구조를 확인하려면 아래 명령을 실행합니다.
```bash
python -m src.train --data_path data/sample_sentiment.csv --epochs 1 --train_batch_size 2 --eval_batch_size 2
```
실제 IMDB 데이터셋을 사용할 경우 CSV 파일에 다음 컬럼이 있어야 합니다.
```text
review,sentiment
영화 리뷰 문장 또는 한국어 리뷰 문장,positive
영화 리뷰 문장 또는 한국어 리뷰 문장,negative
```

실제 데이터 파일을 사용하는 예시는 다음과 같습니다.
```bash
python -m src.train --data_path "data/IMDB Dataset.csv" --epochs 1 --train_batch_size 8 --eval_batch_size 16
```

학습이 끝나면 모델과 토크나이저가 아래 경로에 저장됩니다.
```text
models/bert_sentiment/
```

## 4. Streamlit 앱 실행
```bash
streamlit run app/streamlit_app.py
```

브라우저 화면에서 문장을 입력한 뒤 `감성분석 실행` 버튼을 누르면 다음 결과가 출력됩니다.
- 긍정 또는 부정 분류 결과
- 긍정 확률
- 부정 확률
- 현재 사용한 모델 경로

## 5. 코드 구성 핵심
이 프로젝트는 실제 PyCharm 프로젝트에서 관리하기 쉬운 앱 구조로 분리했습니다.
- `src/data_loader.py`: CSV 파일을 읽고 라벨을 숫자로 변환합니다.
- `src/dataset.py`: 문장을 BERT 입력 텐서로 변환합니다.
- `src/modeling.py`: BERT 분류 모델 생성과 Fine-tuning 전략을 처리합니다.
- `src/train.py`: 학습, 평가, 저장을 실행합니다.
- `src/predict.py`: 저장된 모델을 불러와 문장을 예측합니다.
- `app/streamlit_app.py`: 사용자가 문장을 입력하고 결과를 확인하는 화면입니다.

## 6. 주의사항
기본 모델은  `bert-base-uncased`입니다. 
이 모델은 영어 데이터에 적합합니다. 
한국어 문장 감성분석을 하려면 한국어 데이터셋으로 학습하고, 
`src/config.py`의 `DEFAULT_MODEL_NAME`을 한국어 BERT 모델명으로 바꾸어 사용하는 것이 좋습니다.


## 깃 허브에서 프로젝트 복제한 경우
가상환경 만들기
python -m venv .venv

가상환경 활성화 하기(활성화 안 되었을 경우)
./.venv/Scripts/activate

패키지 설치
pip install -r requirements.txt