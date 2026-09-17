import re

with open('keras/keras36_cnn6_cifar100.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Conv2D 1 comment block
text = re.sub(
    r'# 입력 shape : \(28, 28, 1\).*?마지막 64는 "필터의 개수 = 출력 채널의 개수"이다\.',
    '''# 입력 shape : (32, 32, 3)
#
# 32 : 이미지의 세로 크기
# 32 : 이미지의 가로 크기
# 3  : 채널(channel)의 개수
#
# CIFAR-100은 RGB 컬러 이미지이므로 채널이 3개이다. (32, 32, 3)
#
# --------------------------------------------------------------------
# Conv2D(512, kernel_size=(3,3), padding='same') & MaxPooling2D()
# --------------------------------------------------------------------
#
# 512는 "필터의 개수"이다.
#
# kernel_size=(3,3)은 필터의 가로/세로 크기만 지정한다.
# 실제 필터 하나의 크기는 입력 채널의 깊이까지 포함해서 (3, 3, 3) 이다.
#
# 즉,
# 입력           : (32, 32, 3)
# 필터 하나      : (3, 3, 3)
# 필터 개수      : 512개
#
# padding='same'이므로 Conv2D를 통과해도 가로/세로 크기는 (32, 32)로 유지되며,
# 필터가 512개 있으므로 채널이 512가 된다. -> (32, 32, 512)
#
# 이후 MaxPooling2D()를 통과하면 가로/세로 크기가 절반으로 줄어든다.
#
# 따라서 최종 출력 shape:
#     (16, 16, 512)''',
    text, flags=re.DOTALL
)

# Replace Conv2D 2 comment block
text = re.sub(
    r'# 이전 층의 출력:.*?출력 shape:\n#\n#     \(24, 24, 32\)',
    '''# 이전 층의 출력:
#
#     (16, 16, 512)
#
# 따라서 현재 Conv2D가 받는 입력의 채널 깊이는 512이다.
#
# kernel_size=(3,3)이라고 적었지만 실제 필터 하나의 크기는 (3, 3, 512) 이다.
# 여기서 512는 "입력 채널의 깊이"이다.
#
# 한 위치에서 3 x 3 x 512 = 4,608개의 값을 계산한다.
#
# 필터 하나   : (3, 3, 512)
# 필터 개수   : 256개
#
# padding='same'이므로 가로/세로 크기는 유지된다.
#
# 출력 shape:
#
#     (16, 16, 256)''',
    text, flags=re.DOTALL
)

# Replace Conv2D 3 comment block
text = re.sub(
    r'# 입력:\n#\n#     \(24, 24, 32\).*?출력 shape:\n#\n#     \(22, 22, 32\)',
    '''# 입력:
#
#     (16, 16, 256)
#
# 입력 채널이 256이므로 필터 하나의 실제 크기는 (3, 3, 256)이다.
# 이런 필터가 128개 있다.
#
# padding='same'이므로 크기 유지.
#
# 출력 shape:
#
#     (16, 16, 128)''',
    text, flags=re.DOTALL
)

# Replace Conv2D 4 comment block
text = re.sub(
    r'# 입력:\n#\n#     \(22, 22, 32\).*?출력 shape:\n#\n#     \(20, 20, 16\)',
    '''# 입력:
#
#     (16, 16, 128)
#
# 필터 개수: 64개
#
# 출력 shape:
#
#     (16, 16, 64)''',
    text, flags=re.DOTALL
)

# Replace Conv2D 5 comment block
text = re.sub(
    r'# 입력:\n#\n#     \(20, 20, 16\).*?기존 주석의 \(20,20,16\)은 잘못된 값이다\.',
    '''# 입력:
#
#     (16, 16, 64)
#
# 필터 개수: 8개
#
# 출력 shape:
#
#     (16, 16, 8)''',
    text, flags=re.DOTALL
)

# Replace Dense warning block
text = re.sub(
    r'# 주의!!!.*?즉 현재 모델의 최종 출력 shape는:\n#\n#     \(18, 18, 10\)\n#\n# 이다\.',
    '''# 주의:
# 
# 이전 Conv2D의 출력은 (16, 16, 8) 이었다.
# 이를 Flatten() 층에 통과시키면 1차원 배열로 평탄화된다.
# 
# 16 x 16 x 8 = 2048
# 즉, Flatten() 층의 출력 shape는 (2048,) 이 된다.
# 
# 그 후 Dense 층들을 거치며 최종적으로 100개의 클래스에 대한 확률을 출력한다.
# 최종 출력 shape: (100,)''',
    text, flags=re.DOTALL
)

# Replace Dropout blocks
text = re.sub(r'# 입력  : \(24, 24, 32\)\n# 출력  : \(24, 24, 32\)', r'# 입력: (16, 16, 256)\n# 출력: (16, 16, 256)', text)
text = re.sub(r'# 입력  : \(20, 20, 16\)\n# 출력  : \(20, 20, 16\)', r'# 입력: (16, 16, 64)\n# 출력: (16, 16, 64)', text)

# Replace final evaluation y_pred.shape
text = text.replace('# (10000, 18, 18, 10)', '# (10000, 100)')

with open('keras/keras36_cnn6_cifar100.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done CIFAR100")
