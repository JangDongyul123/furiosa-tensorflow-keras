# Furiosa TensorFlow · Keras Study

Furiosa AI Agent 과정에서 학습한 TensorFlow/Keras 수업 필기와 실습 코드를 정리한 저장소입니다.

## 학습 내용

- TensorFlow/Keras 기본 사용법
- Dense Layer와 다층 신경망
- 회귀, 이진 분류, 다중 분류
- Train, Validation, Test 데이터 분리
- R², RMSE, Accuracy, ROC-AUC 평가 지표
- 과적합과 EarlyStopping
- MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
- 모델과 가중치 저장 및 불러오기
- ModelCheckpoint
- Dropout과 Functional API
- CNN 기초와 MNIST 데이터

## 필기 목차

| 일차 | 주요 내용 | 필기 | 관련 Velog 글 |
|---:|---|---|---|
| 1일차 | 개발 환경, Batch, 신경망, 가중치와 편향, 역전파, Loss | [1일차 필기](./keras/1일차%20필기.md) | - |
| 2일차 | 스칼라·벡터·행렬·텐서, 행렬곱, Shape, Sequential 모델 | [2일차 필기](./keras/2일차%20필기.md) | - |
| 3일차 | Train/Test 데이터 분리, `train_test_split`, MSE | [3일차 필기](./keras/3일차%20필기.md) | - |
| 4일차 | 데이터의 중요성, 회귀와 분류, 데이터 전처리 | [4일차 필기](./keras/4일차%20필기.md) | [MSE·RMSE·RMSLE·R²의 관계](https://velog.io/@wordi/MSE-RMSE-RMSLE-R2%EC%9D%98-%EA%B4%80%EA%B3%84) |
| 5일차 | Dense Layer, ReLU, Hidden Layer와 Output Layer | [5일차 필기](./keras/5일차%20필기-%20ReLU.md) | [초평면·뉴런·ReLU·순전파와 역전파](https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C) |
| 6일차 | Validation, 과적합, History 시각화, EarlyStopping | [6일차 필기](./keras/6일차%20필기.md) | - |
| 7일차 | 학습 시간 측정, 과적합 방지, 이진 분류 | [7일차 필기](./keras/7일차%20필기.md) | [MSE와 Cross-Entropy 비교](https://velog.io/@wordi/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8%EC%97%90%EC%84%9C-loss-%ED%95%A8%EC%88%98-%EC%84%B1%EB%8A%A5-%EB%B9%84%EA%B5%90-Mean-Squared-Error-VS-Cross-Entropy) |
| 8일차 | 회귀, 이진 분류, 다중 분류 비교 | [8일차 필기](./keras/8일차%20필기.md) | - |
| 9일차 | Softmax, One-Hot Encoding, Argmax, ROC-AUC, Input Shape | [9일차 필기](./keras/9일차%20필기.md) | - |
| 10일차 | 여러 Scaler 비교, 이상치, 모델 저장 | [10일차 필기](./keras/10일차%20필기.md) | - |
| 11일차 | Sequential·Functional 모델, Dropout, Branch, Ensemble | [11일차 필기](./keras/11일차%20필기.md) | [Dropout을 아주 쉽게 이해하기](https://velog.io/@wordi/Dropout%EC%9D%84-%EC%95%84%EC%A3%BC-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0) |

## 학습 내용 예시

### 입력 Feature 수와 초평면

입력 Feature가 1개이면 선형 모델은 직선으로, 2개이면 평면으로 표현됩니다.

[![입력 Feature 수에 따른 직선과 평면](./images/hyperplane-feature.png)](https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C)

[관련 Velog 글 읽기](https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C)

## 실습 코드 구성

| 파일 범위 | 내용 |
|---|---|
| `keras01.py` ~ `keras08_*.py` | Keras 기초, Dense, Batch, 행렬, 다중 입력과 출력 |
| `keras09_*.py` ~ `keras20_*.py` | 데이터 분리, 회귀, 평가 지표, Validation, 과적합, EarlyStopping |
| `keras21_*.py` ~ `keras28_*.py` | 이진·다중 분류, Input Shape, Scaling |
| `keras29_*.py` ~ `keras32_*.py` | 모델 저장과 불러오기, ModelCheckpoint |
| `keras33_*.py` | Dropout |
| `keras34_*.py` | Functional API |
| `keras35_*.py` | GPU 실행 테스트 |
| `keras36_*.py` | CNN과 MNIST |

## 사용한 데이터셋

| 문제 유형 | 데이터셋 |
|---|---|
| 회귀 | California Housing, Diabetes, Boston Housing, Dacon 따릉이, Kaggle Bike Sharing |
| 이진 분류 | Breast Cancer Wisconsin, Santander Customer Transaction |
| 다중 분류 | Iris, Wine, Covertype, Digits, MNIST |

Dacon과 Kaggle 실습 데이터는 저장소에 포함되어 있지 않습니다. 코드를 실행하려면 각 파일에서 사용하는 `./_data/...` 경로에 데이터를 준비해야 합니다.

## 개발 환경

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib

필요한 패키지는 다음과 같이 설치할 수 있습니다.

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

일부 코드는 로컬 데이터 또는 저장된 모델의 경로를 사용하므로 실행 전에 파일 안의 `path` 값을 확인해야 합니다.

## 블로그

수업 내용과 공부하면서 궁금했던 내용을 Velog에도 정리하고 있습니다.

[AI 엔지니어링 학습노트](https://velog.io/@wordi/series/AI-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%ED%95%99%EC%8A%B5%EB%85%B8%ED%8A%B8)

현재 저장소의 수업 진도와 직접 연결되지는 않지만, 딥러닝을 공부하며 생각한 내용도 별도로 기록했습니다.

- [딥러닝을 공부하며 깨달은 프롬프트 엔지니어링의 본질](https://velog.io/@wordi/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%84-%EA%B3%B5%EB%B6%80%ED%95%98%EB%A9%B0-%EA%B9%A8%EB%8B%AC%EC%9D%80-%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81%EC%9D%98-%EB%B3%B8%EC%A7%88)
