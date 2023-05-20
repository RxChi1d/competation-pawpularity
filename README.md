- 模仿Kaggle的資料夾結構，如下：  
```
├── requirements.txt
├── input
│   ├── petfinder-pawpularity-score
└── working
    ├── lightning_logs
    ├── train.ipynb
    └── test.ipynb
```
- 運行的時候，請將工作目錄位置設在working

- 訓練時，請按需求仔細調整以下兩者：
    1. config  
    2. import所需的model  

    理論上其他的東西應該不需要更改  

- 測試時，請上傳你的權重到kaggle上，並import test.ipynb作為測試用code，需要調整save_dir成你上傳的dataset的「名稱」（注意不是路徑）