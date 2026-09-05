# 5일차 필기- ReLU

## 1. Dense Layer의 기본 연산

Dense Layer의 기본 계산은 다음과 같이 표현할 수 있다.

```text
z = WX + B
```

입력 피처가 여러 개라면 하나의 뉴런은 다음과 같은 선형 결합을 계산한다.

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

- `X` : 입력값(Input Feature)
- `W` : 가중치(Weight)
- `B` : 편향(Bias)
- `z` : 활성화 함수 적용 전 값

활성화 함수 없이 Dense Layer만 여러 층 연결하면 전체 변환은 다시 하나의 선형 변환으로 정리될 수 있다.

---

## 2. ReLU

ReLU(Rectified Linear Unit)는 대표적인 활성화 함수(Activation Function)이다.

```text
ReLU(z) = max(0, z)
```

즉,

```text
z > 0  -> z
z <= 0 -> 0
```

예:

```text
ReLU(5)  = 5
ReLU(-3) = 0
```

따라서 ReLU는 `WX+B`가 음수이면 0으로 만들고, 양수면 그대로 통과시킨다.

---

## 3. ReLU를 사용하는 이유

ReLU의 역할은 단순히 음수를 제거하는 데 그치지 않는다.

입력값에 따라 뉴런의 출력이

```text
0
```

또는

```text
WX+B
```

로 달라지면서 모델 내부에 서로 다른 선형 구간이 만들어질 수 있다.

---

## 4. Hidden Layer와 Output Layer

Hidden Layer에서 ReLU를 사용하더라도 최종 출력값이 반드시 0 이상이 되는 것은 아니다.

마지막 Dense Layer가 다음과 같고:

```text
y_hat = w1*h1 + w2*h2 + ... + b
```

출력층에 ReLU를 사용하지 않는다면, 가중치와 편향에 따라 최종 `y_hat`은 음수가 될 수도 있다.

---

## 5. Output Layer에 ReLU 적용을 지양하는 이유

출력층에 ReLU를 적용하면 모델의 출력값이 0 미만으로 내려가지 못해 출력 범위가 제한된다. 
타겟 데이터가 음수를 가질 수 있는 일반적인 상황에서는 모델의 표현력을 심각하게 왜곡하므로, 
출력층에는 특별한 도메인 제약(출력이 반드시 양수여야 하는 경우 등)이 없는 한 ReLU를 적용하지 않는다.
