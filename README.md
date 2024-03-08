# Model Interpretability

## 環境構築(Docker)

### コンテナの作成と仮想環境の起動

1. コンテナの作成と実行

```
docker compose up -d
```

2. コンテナのシェルを起動する

```
docker compose exec -it model_interpretability /bin/bash
```

3. シェルから抜ける

```
exit
```

### コンテナの停止

```
docker compose stop
```

再起動する際は以下のコマンドを実行する。

```
docker compose start
```

### コンテナの削除

```
docker compose down
```

## 特徴量寄与度の可視化

1\. [config.yaml]("./config/config.yaml)を編集する

2\. コンテナを起動した状態でコマンドを実行する

```
python src/<ファイル名 (eg. gradcam.py)>
```

※ `gradcam.py`と`guided_gradcam.py`の場合は`layer`の指定が必要
