關鍵字:
說中文關鍵字辨識
['向後' ,'go', '往左', '往右', '停' ,'向前']

原始檔來源參考以下，請自行調整model參數
https://www.tensorflow.org/tutorials/audio/simple_audio
訓練資料:每分類['向後' ,'go', '往左', '往右', '停' ,'向前']錄音檔(各1秒)存為wav格式
ps:這測試每各關鍵字項目各200個wav檔案(各1秒)


1->於train/app.py中訓練自已的聲音檔案
2->執行app01.py和app02.py載入saved models執行測試