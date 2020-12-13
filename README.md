# Realtime ObjectDetection with flask
pytorchとflaskを使用した物体検出webアプリケーションです。  
[https://github.com/qfgaohao/pytorch-ssd] をもとにしています。

## 環境構築
1. Anacondaをダウンロードし、実行権限を付与して実行
    ```
    $ wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    $ $ ./Anaconda3-2020.07-Linux-x86_64.sh
    $ conda upgrade --all
    $ conda clean -packages
    ```

1. 環境を作成
    ```
    $ conda env create -f myenv.yaml
    ```

    上記コマンドで環境が作成出来ない場合、requirements.txtから必要なパッケージのインストールを行ってください。


## 訓練
1. ```vision/utils/config```内のroot_dirを環境に合わせて書き換えてください。

1. VOC2012データセットのダウンロード
    ```
    $ python download_data.py
    ```
    ```data```ディレクトリにVOC2012データセットがダウンロードされます。  
    通信環境によりことなりますが、私の場合ダウンロードに20分ほどかかりました。

1. train.pyを実行
    ```
    $ python train.py --datasets data/VOCdevkit/VOC2012 
    ```
    エポック数やバッチサイズ等のハイパーパラメータは、```vision/utils/config.py```内にて変更できます。


## web上で物体検出
```
$ cd app/
$ python app.py
```
ブラウザが立ち上がり、物体検出アプリが開きます。何も表示されない場合、リロードしてください。

capture startボタンを押すことにより、録画開始ポイントをリセット、save captureボタンを押すことにより、録画開始ポイントからsave captureボタンを押すまでの動画を出力することが出来ます。
## DIR
最終的なディレクトリ構成を以下に示します。
```
${root}
|-- app
|   |-- app.py
|   |-- templates
|   |   |-- index.html
    |-- output
|-- data
|   |-- VOCdevkit
|   |   |-- VOC2012
|-- models
|   |-- mb2-imagenet-71_8.pth
|   |-- mb2-ssd-lite-Epoch-149-Loss-2.6.5.pth //私のPCで訓練したモデルです。
|   |-- voc-model-labels.txt
|-- vision
|   |-- nets
|   |   |-- mobilenet_v2_ssd_lite.py
|   |   |-- mobilenet_v2.py 
|   |   |-- ssd.py
|   |   |-- predictor.py
|   |-- utils
|   |   |-- box_utils.py
|   |   |-- config.py
|   |   |-- data_preprocessing.py
|   |   |-- misc.py
|   |   |-- multibox_loss.py
|   |   |-- transforms.py
|   |   |-- voc_dataset.py
|-- download_data.py 
|-- train.py
|-- README.md
```
