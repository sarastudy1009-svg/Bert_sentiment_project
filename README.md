# BERT Sentiment Analysis Streamlit Project

BERT fine-tuning 코드를 `src/`에 분리하고, Streamlit 앱에서 리뷰 문장을 입력하면 긍정/부정 감정분석 결과를 출력하는 프로젝트입니다.

## 1. 프로젝트 구조

```text
Bert_sentiment_project/
├─ app/
│  └─ streamlit_app.py          # Streamlit 화면 실행 파일
├─ data/
│  ├─ ratings_train.txt         # 한국어 데이터
│  └─ sample_sentiment.csv      # 실행 테스트용 예제 데이터
├─ models/
│  └─ .gitkeep                  # 학습 모델 저장 폴더
│
├─ src/                         # [영어]
│  ├─ __init__.py
│  ├─ config.py                 # 경로, 모델명, 라벨 설정
│  ├─ data_loader.py            # CSV 로드 및 train/valid/test 분리
│  ├─ dataset.py                # BERT 입력용 PyTorch Dataset
│  ├─ modeling.py               # BERT 모델 생성 및 fine-tuning 전략
│  ├─ predict.py                # 문장 감정 예측 클래스
│  ├─ train.py                  # 모델 학습 실행 스크립트
│  └─ utils.py                  # seed 고정, device 선택 함수
│
├─ src_kweon_sora/              # [한국어]
│  ├─ __init__.py
│  ├─ config.py                 # 경로, 모델명, 라벨 설정
│  ├─ data_loader.py            # CSV 로드 및 train/valid/test 분리
│  ├─ dataset.py                # BERT 입력용 PyTorch Dataset
│  ├─ modeling.py               # BERT 모델 생성 및 fine-tuning 전략
│  ├─ predict.py                # 문장 감정 예측 클래스
│  ├─ train.py                  # 모델 학습 실행 스크립트
│  └─ utils.py                  # seed 고정, device 선택 함수
│
├─ requirements.txt             # 설치 패키지 목록
├─ .gitignore                   # Git 제외 파일 설정
├─ 실습문제.md                  # 실습 요구사항
└─ README.md                    # 실행 방법 설명
```

`src_kweon_sora/`는 `src/`와 같은 구조이며, 내부 import만 `src_kweon_sora` 패키지를 바라보도록 분리되어 있습니다.

## 2. 실행 준비

Python 3.11 가상환경을 권장합니다.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3. 모델 학습

기본 예제 데이터로 구조를 빠르게 확인하려면 다음 명령을 실행합니다.

```bash
python -m src.train --data_path data/sample_sentiment.csv --epochs 1 --train_batch_size 2 --eval_batch_size 2
```

개인 브랜치용 패키지를 실행하려면 다음 명령을 사용합니다.

```bash
python -m src_kweon_sora.train --data_path data/sample_sentiment.csv --epochs 1 --train_batch_size 2 --eval_batch_size 2
```

CSV 파일은 다음 컬럼을 가져야 합니다.

```text
review,sentiment
This movie was wonderful.,positive
The film was boring.,negative
```

학습이 끝나면 모델과 토크나이저가 아래 경로에 저장됩니다.

```text
models/bert_sentiment/
```

## 4. Streamlit 실행

```bash
streamlit run app/streamlit_app.py
```

브라우저 화면에서 문장을 입력하고 감정분석 버튼을 누르면 다음 결과가 출력됩니다.

- 긍정 또는 부정 분류 결과
- 긍정 확률
- 부정 확률
- 현재 사용 중인 모델 경로

## 5. 브랜치 작업 및 업로드

현재 브랜치 확인:

```bash
git branch --show-current
```

작업 내용을 커밋하고 현재 브랜치에 올리기:

```bash
git add .
git commit -m "Add kweon_sora sentiment package"
git push
```

대용량 데이터 파일을 GitHub에 올리지 않을 경우 `.gitignore`에 해당 파일을 추가하고 추적을 해제합니다.

```bash
git rm --cached data/ratings_train.txt
```
