# 資料夾結構  
- 模仿Kaggle的資料夾結構，如下：  
    ```
    ├── requirements.txt
    ├── input
    │   ├── petfinder-pawpularity-score
    └── working
        ├── dataset_temp.pickle
        ├── predict.ipynb
        ├── test-ensemble-byhand-final.ipynb
        ├── test.ipynb
        └── train.ipynb
    ```
- 運行的時候，請將工作目錄位置設在working  

# 下載 petfinder-pawpularity-score dataset  
- Source: https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview
- 透過 Kaggle API 下載  
    ```
    kaggle competitions download -c petfinder-pawpularity-score
    ```

# 檔案說明
- train.ipynb: 訓練
    - 訓練前，請按需求仔細調整以下兩者：  
        1. config  
        2. import 需要的 model  
    - 理論上其他的東西應該不需要更改  
- test.ipynb: 測試，並輸出模型的 mean RMSE  
- predict.ipynb: 用於生成submission.csv，於 kaggle 上進行測試
    - 測試時，請上傳你的權重到kaggle上，並 import predict.ipynb 作為測試用code，需要調整save_dir成你上傳的dataset的「名稱」（注意不是路徑）
- test-ensemble-byhand-final.ipynb: 在 kaggle 上用來對不同模型進行加權平均，final 表示檔案中的權重生成了最好的結果
- dataset_temp.pickle: 5 fold 的資料集分割暫存檔，用來記錄不同 fold 中 train 與 val 對應的 index

# 最終結果
本次實驗最終於 kaggle 上得到之 Private Score 為 16.95695。  