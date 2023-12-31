XAI OCR 
========
2조

김민지

이동준

전가위

최종인

---
Easy OCR, Tesseract OCR 사용
데모 영상 파일 있음

---

# 수많은 고난의 과정 (정리 자료)
[Config instructions for the PSENet text detection module in MMOCR..pdf](./additional_data/Config%20instructions%20for%20the%20PSENet%20text%20detection%20module%20in%20MMOCR..pdf)

[K-means clustering.pdf](./additional_data/K-means%20clustering.pdf)

[KNN](./additional_data/KNN.pdf)

[Tesseract.pdf](./additional_data/Tesseract.pdf)

# 우리는 어쩌면 좋을까? (회의록)

### 20231004
    
    ### 확인사항
    
    - 큰 그림을 우선 그리는 것이 목표이다. 모두 이 분야가 전공이 아니기 때문에, 큰 그림을 그리고 최대한 쉬운 방법부터 차근차근 접근하는 것이 중요해보인다.
    
    ### 회의 내용
    
    - 각자 조사한 내용을 발표하였다. 크게 이미지나 문서에서 글자가 있는 부분을 인식하는 Text Detection 부분과, 글자가 있는 부분에서 실제 글자를 인식하여 출력하는 Text Recognition 부분으로 분리할 수 있다. 그리고, 두 가지 역할을 모두 수행하는 모델이 있고 (End-to-End), 각각에 집중하는 모델들이 있다.
    - 하지만, 문제상황에 맞게 Text Detection과 Recognition을 제외한 후처리 부분을 만들기로 결정했다. 이를 통해, 단순히 인식만 하여 끝내는 것이 아닌 어느 부분에 들어와야 하는지, 무엇을 의미하는지 조금 더 자세히 알 수 있는 최종 모델을 만들기로 결정했다.
    
    ### 다음 주 목표
    
    - 우선, Text Detection 모델과 Text Recognition의 모델들의 종류를 알아온 뒤, 우리가 직접 모두 구현해야 하는지, 아니면 기존에 있는 모델들을 모아놓은 git repository 등이 있는지 확인해보기로 했다.
### 20231010
    
    ### 확인사항
    
    - 각 모델에 대한 종류를 알아온 뒤, 모아놓은 OCR을 사용할지 코드를 가지고 직접 구현할지 결정하기로 하였다.
    
    ### 회의 내용
    
    - 굉장히 많은 종류의 OCR 기법들이 있었다.
    - 이런 OCR 기법들을 모아놓은 OCR 친구들도 있었다. 실험을 쉽게 하는 것이 목적인 것 같다.
    - mmOCR 등을 사용해도 괜찮을지에 대한 의견이 있었다.
    
    ### 다음 주 목표
    
    - 각자 조금 더 알아보기로 했다. mmOCR을 사용해도 좋을지, 직접 코드를 가지고 구현해야 할 지에 대해 조금 더 생각해보기로 하였다.
### 20231017
    
    ### 확인사항
    
    - 코드만 나와있는 두 모델을 합치는게 나을지, 코드로 많은 모델들을 모아놓은 mmOCR등을 사용하는게 좋을지 파악해보기로 했다.
    
    ### 회의 내용
    
    - mmOCR의 코드를 조금 분석해 본 결과, 충분히 많은 fine tuning이 가능할 것으로 파악되어 mmOCR의 Text Detection, Recognition 모델을 사용하기로 결정했다.
    - 이제 mmOCR의 어떤 모델을 사용하는게 좋을지 결정한 후 중간 발표 슬라이드를 제작하면 된다.
    
    ### 다음 주 목표
    
    - 각자 어떤 모델을 사용하는게 좋을지 파악한 후, 중간 발표 슬라이드를 만들면 된다. 마음에 드는 모델을 하나씩 골라서 단체 채팅방에 업로드 하기로 했다.
    - PPT의 경우, 발표기간에 학회에 가있어 즉각적인 피드백이 불가능한 이동준/최종인 두 명이 만들고, 이외에 녹화/피드백은 김민지/전가위 님이 맡아주기로 하였다.
### 20231107
    
    ### 확인사항
    
    - 각자 다른 조의 발표를 보고 배울만한 점이 있는지 확인해보기로 했다.
    - 데이터를 주시는지 확인해야 한다.
    
    ### 회의 내용
    
    - 데이터는 아직 말씀이 없으시다. 데이터를 주시지 않는다고 하면, 우리가 처음 생각한 후처리 모델은 만들기 어려울 수 있다. 지도학습 기반의 후처리 모델이 될 수도 있어서, 데이터가 먼저 필요하다.
    - 기말 준비를 해야한다. 우선, 각각의 Text Detection, Recognition 모델을 사용하기 때문에, 두 모델이 호환이 가능한지 (input/output형식이 같은지) 확인해보아야 한다.
    
    ### 다음 주 목표
    
    - 각각 Text Detection 모델로 선정한 PSENet의 output, Text Recognition 모델로 선정한 SATRN의 input이 동일한지 살펴보기로 했다.
### 20231114
    
    ### 확인사항
    
    - PSENet의 output과 SATRN의 input 형식이 동일해야 두 모델을 사용할 수 있기 때문에, 각각의 input과 output을 알아야 한다.
    - 데이터를 받을 수 있는지 확인해야 한다.
    
    ### 회의 내용
    
    - 데이터를 이번주 안으로 주신다고 했기 때문에, 데이터를 받아보고 어떻게 사용할 수 있을지 확인해봐야 한다.
    - 이미지인지 pdf인지, 각 형식에 따른 정답도 주는지 혹은 파일만 주는지 등에 따라 후처리(기계학습을 통한 fine tuning)부분을 제대로 할 수 있다.
    - 데이터를 받은 다음 주부터 후처리 모델을 만들어야 하기 때문에, 최대한 이번주에 Text Detection + Text Recognition 모델에 대한 학습 및 사용이 완료되어야 한다.
    - SATRN 모델의 output이 단순히 text인 것으로 보이는데, 우리는 후처리를 위해 (각 text가 어떤 속성에 들어가는지 파악하고 연결하기 위해) 좌표도 살려놓아야 한다. SATRN의 output을 조금 수정해야 할 수도 있다.
    - 후처리용 기계학습 모델은 AOT 혹은 Decision Tree로 결정되었다. 이 두 모델은 XAI를 적용/흉내내기 좋아서 결정하였다.
    
    ### 다음 주 목표
    
    - OCR 모델을 합쳐야 한다 (PSENet + SATRN). 이와 동시에 SATRN 모델의 output의 형식을 (좌표 + text)로 바꿔보기로 했다.
    - 제공해주시는 order form 자료가 어떤 형식인지에 따라 다르다. 만약 파일만 주신다면, 최대한 다음 회의까지 정답을 만들어가기로 했다. 정답을 만들어야 후처리용 기계학습 모델을 만들기 편하다.
### 20231128
    
    ### 확인사항
    
    - 데이터베이스 자료형 확인
    - OCR 모델 합칠 수 있는지 확인, 좌표 출력 여부 확인
    
    ### 회의 내용
    
    - 데이터가 올라왔는데, order form이 단순히 pdf의 형태였다. 이를 직접 labeling 작업하거나 만들 수 있는 방법을 찾아야 한다.
    - OCR 모델을 합치는데는 성공했지만, 받은 데이터와의 궁합이 좋지 않다. 좌표는 출력 성공했다.
    - 생각보다 성능이 좋지 않아, 다른 방법을 찾아봐야 할 것 같다.
    
    ### 다음 주 목표
    
    - 다른 OCR 모델이 있는가?
    - labeling을 어떻게 수행할 것인가?
### 20231205
    
    ### 확인사항
    
    - 다른 OCR 모델이 있는가?
    - labeling을 어떻게 수행할 것인가?
    
    ### 회의 내용
    
    - Easy OCR, TesseractOCR을 사용하기로 했다. 쓰기 편하고, 성능도 유난히 좋아서 이 두 가지를 모두 사용하여 서로 보완하기로 했다.
    - labeling의 경우, 우선 보류하기로 했다. 너무 많았고, 각 단어가 무슨 의미인지 몰라 잘못된 결과가 나올 수 있을 것 같았다.
    - 우선 EasyOCR, TesseractOCR을 사용하여 중요한 정보인 date, destination 등을 제대로 파악할 수 있는 방법을 생각해보기로 했다.
    
    ### 다음 주 목표
    
    - date, destination 정확하게 판단하는 방안 마련