# KNU 딥러닝 특강 [2019.02.25~28]

## [2019.02.25] Python & numpy

[딥러닝(Deep-Learning)을 위한 프로그램 설치](https://m.post.naver.com/my/series/detail.nhn?memberNo=8098532&seriesNo=459452&prevVolumeNo=15102526)

아래 링크의 포스트를 순서대로 따라가면서 그래픽 드라이버, CUDA, CUDNN, Anaconda, tensorflow 를 설치한다.
내 노트북의 그래픽카드는 GeForce MX150 이므로 tensorflow_gpu-1.12.0 을 설치하기 위해 CUDA 9.0을 설치하고, 이에 맞추어 CUDNN 7.4.2의 라이브러리를 다운받아 추가하였다.
또한 CUDA 9.0이 Python 3.7을 지원하지 않으므로 Python의 버전을 3.6.8로 다운그레이드하였다.
가상환경 상의 파이썬에서 tensorflow를 구동시키기 위하여 Anaconda3을 설치한다.


### Colab
구글에서 제공하는 Jupyter notebook 기반의 온라인 딥러닝 구동 환경이다. 구글 드라이브, Github 등과 연동하여 사용할 수 있어 편리하다.
교육, 학습용으로 제공되는 서비스이기 때문에 일정 시간이상 딥러닝을 구동시키면 실행이 초기화된다.
하지만 이는 tensorflow의 기본적인 기능을 학습하는 데에 큰 지장은 없다.



## [2019.02.26] MLP

MLP : Mulitlayered Perceptron

### Iris

## [2019.02.27] CNN

CNN : Convolutional Neural Network

### MNIST 


![](images/MNIST_CNN.PNG)


## [2019.02.28] Keras

