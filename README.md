<div align="center">

# 📘 Furiosa TensorFlow · Keras Study

TensorFlow/Keras 기반 딥러닝 수업 필기와 실습 기록

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
![Study Notes](https://img.shields.io/badge/Study_Notes-11-4C566A?style=flat-square)
![Practice Files](https://img.shields.io/badge/Practice_Files-125-5E81AC?style=flat-square)

[📚 학습 필기](#-학습-필기-study-notes) · [💻 실습 구성](#-실습-구성-practice-structure) · [✍️ Velog](https://velog.io/@wordi/series/AI-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%ED%95%99%EC%8A%B5%EB%85%B8%ED%8A%B8)

</div>

---

## 📌 개요 (Overview)

Furiosa AI Agent 과정에서 학습한 TensorFlow/Keras 내용을 정리한 저장소입니다.  
수업 필기와 실습 코드를 날짜와 주제에 따라 기록하고 있습니다.

| 구분 | 학습 범위 |
|---|---|
| 신경망 기초 | Dense Layer, Weight, Bias, Forward Propagation, Backpropagation |
| 문제 유형 | Regression, Binary Classification, Multi-class Classification |
| 데이터 처리 | Train/Validation/Test Split, Scaling, Input Shape |
| 모델 학습 | Loss Function, Optimizer, Batch, Epoch |
| 평가 | R², RMSE, Accuracy, ROC-AUC |
| 일반화 | Overfitting, EarlyStopping, Dropout |
| 모델 관리 | Model Save/Load, Weight Save/Load, ModelCheckpoint |
| 모델 구조 | Sequential API, Functional API, CNN |

### 학습 흐름

```mermaid
flowchart LR
    A["🧠 신경망 기초"] --> B["📈 회귀"]
    B --> C["🎯 분류"]
    C --> D["🛠️ 모델 개선"]
    D --> E["🖼️ CNN"]
```

---

## 📚 학습 필기 (Study Notes)

| 일차 | 주요 내용 | 필기 | 관련 Velog 글 |
|---:|---|---|---|
| 1일차 | 개발 환경, Batch, 신경망, 가중치와 편향, 역전파, Loss | [필기 보기](./keras/1일차%20필기.md) | — |
| 2일차 | 스칼라·벡터·행렬·텐서, 행렬곱, Shape, Sequential 모델 | [필기 보기](./keras/2일차%20필기.md) | — |
| 3일차 | Train/Test 데이터 분리, `train_test_split`, MSE | [필기 보기](./keras/3일차%20필기.md) | — |
| 4일차 | 데이터의 중요성, 회귀와 분류, 데이터 전처리 | [필기 보기](./keras/4일차%20필기.md) | [MSE·RMSE·RMSLE·R²의 관계](https://velog.io/@wordi/MSE-RMSE-RMSLE-R2%EC%9D%98-%EA%B4%80%EA%B3%84) |
| 5일차 | Dense Layer, ReLU, Hidden Layer와 Output Layer | [필기 보기](./keras/5일차%20필기-%20ReLU.md) | [초평면·뉴런·ReLU·순전파와 역전파](https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C) |
| 6일차 | Validation, 과적합, History 시각화, EarlyStopping | [필기 보기](./keras/6일차%20필기.md) | — |
| 7일차 | 학습 시간 측정, 과적합 방지, 이진 분류 | [필기 보기](./keras/7일차%20필기.md) | [MSE와 Cross-Entropy 비교](https://velog.io/@wordi/%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8%EC%97%90%EC%84%9C-loss-%ED%95%A8%EC%88%98-%EC%84%B1%EB%8A%A5-%EB%B9%84%EA%B5%90-Mean-Squared-Error-VS-Cross-Entropy) |
| 8일차 | 회귀, 이진 분류, 다중 분류 비교 | [필기 보기](./keras/8일차%20필기.md) | — |
| 9일차 | Softmax, One-Hot Encoding, Argmax, ROC-AUC, Input Shape | [필기 보기](./keras/9일차%20필기.md) | — |
| 10일차 | MinMax·Standard·MaxAbs·Robust Scaler, 이상치, 모델 저장 | [필기 보기](./keras/10일차%20필기.md) | — |
| 11일차 | Sequential·Functional 모델, Dropout, Branch, Ensemble | [필기 보기](./keras/11일차%20필기.md) | [Dropout을 아주 쉽게 이해하기](https://velog.io/@wordi/Dropout%EC%9D%84-%EC%95%84%EC%A3%BC-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%98%EA%B8%B0) |

---

## 🖼️ 학습 내용 예시 (Visual Note)

### 입력 Feature 수와 초평면

입력 Feature가 1개인 선형 모델은 2차원 공간의 직선으로, 입력 Feature가 2개인 선형 모델은 3차원 공간의 평면으로 표현됩니다.

<p align="center">
  <a href="https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C">
    <img src="./images/hyperplane-feature.png" alt="입력 Feature 수에 따른 직선과 평면" width="820">
  </a>
</p>

<p align="center">
  <sub>ChatGPT를 활용해 제작한 학습용 이미지 · 내용 구성 및 검수: 작성자</sub>
</p>

자세한 내용은 [초평면·뉴런·ReLU·순전파와 역전파](https://velog.io/@wordi/%EA%B7%80-%EC%B4%88%ED%8F%89%EB%A9%B4-%EB%89%B4%EB%9F%B0-ReLU-%EC%88%9C%EC%A0%84%ED%8C%8C%EC%99%80-%EC%97%AD%EC%A0%84%ED%8C%8C)에서 확인할 수 있습니다.

---

## 💻 실습 구성 (Practice Structure)

코드 파일은 학습 순서에 따라 번호를 붙여 관리합니다.

| 파일 범위 | 내용 |
|---|---|
| `keras01.py` – `keras08_*.py` | Keras 기초, Dense, Batch, 행렬, 다중 입력과 출력 |
| `keras09_*.py` – `keras20_*.py` | 데이터 분리, 회귀, 평가 지표, Validation, 과적합, EarlyStopping |
| `keras21_*.py` – `keras28_*.py` | 이진·다중 분류, Input Shape, Scaling |
| `keras29_*.py` – `keras32_*.py` | 모델 저장과 불러오기, ModelCheckpoint |
| `keras33_*.py` | Dropout |
| `keras34_*.py` | Functional API |
| `keras35_*.py` | GPU 실행 테스트 |
| `keras36_*.py` | CNN과 MNIST |

---

## 🗂️ 데이터셋 (Datasets)

| 문제 유형 | 데이터셋 |
|---|---|
| 회귀 | California Housing, Diabetes, Boston Housing, Dacon 따릉이, Kaggle Bike Sharing |
| 이진 분류 | Breast Cancer Wisconsin, Santander Customer Transaction |
| 다중 분류 | Iris, Wine, Covertype, Digits, MNIST |

> Dacon과 Kaggle 실습 데이터는 저장소에 포함되어 있지 않습니다.  
> 코드를 실행하려면 각 파일에서 사용하는 `./_data/...` 경로에 데이터를 준비해야 합니다.

---

## ⚙️ 개발 환경 (Environment)

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

일부 코드는 로컬 데이터 또는 저장된 모델 경로를 사용합니다. 실행하기 전에 파일 안의 `path` 값을 현재 환경에 맞게 수정해야 합니다.

---

## ✍️ 관련 글 (Related Writing)

- [AI 엔지니어링 학습노트 전체 보기](https://velog.io/@wordi/series/AI-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%ED%95%99%EC%8A%B5%EB%85%B8%ED%8A%B8)
- [딥러닝을 공부하며 깨달은 프롬프트 엔지니어링의 본질](https://velog.io/@wordi/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%84-%EA%B3%B5%EB%B6%80%ED%95%98%EB%A9%B0-%EA%B9%A8%EB%8B%AC%EC%9D%80-%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81%EC%9D%98-%EB%B3%B8%EC%A7%88)
