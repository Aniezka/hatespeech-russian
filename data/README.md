# Data and Manipulation

## STEP 1
- work_out_threads_2ch.ipynb - created [a dataset of the form Comment - Reply](https://disk.yandex.ru/d/RhtQtoIvWrvN5Q); preprocessing of [the 2ch dataset](https://disk.yandex.ru/d/08SWZXGSS2c-mQ). Concatenated data from 4 datasets: prodota.ru, 2ch, open subtitles, fiction.
-	subtitles.ipynb – code to create [a dataset of subtitles](https://disk.yandex.ru/d/8x7m_0a3fDapaQ) created from https://disk.yandex.ru/d/TPraG0bLP6UuVw 
- prodota_crauler.ipynb - crauler of prodota.ru 
- prodota_clean.ipynb - preprocessing of [a dataset from prodota.ru](https://disk.yandex.ru/d/SP-VSomE_fhJgQ)
- classic_literature.ipynb - code to create a dataset of fiction from https://raw.githubusercontent.com/Koziev/NLP_Datasets/master/Conversations/Data/ru.conversations.txt
- load_threads_2ch
  -	all_boards.txt - the boards of 2ch we took
  -	load_threads_2ch.ipynb – the code to save the threads

Link to data: https://disk.yandex.ru/d/Xe52jZLB1n3acA 

## STEP 2
- preprocessing_dataset.ipynb - preprocessing of the created dataset on the STEP 1

Link to data: https://disk.yandex.ru/d/wt_YflsLYl95PQ

## STEP 3
- toxicity_classifier.ipynb - code with toxicity classifier
- concatenate_datasets.ipynb - restore the original Comment/Reply for each pair of preprocessed Comment/Reply. 

Link to data: https://disk.yandex.ru/d/v9XIZwPKrX_Hgw

## STEP 4
- key_words_match.ipynb – code to extract identifiers of hate speech.
- concat_key_words.ipynb – code to concatenate lists (Comment/Reply) of identifiers of hate speech for each target group.

Link to data: https://disk.yandex.ru/d/I6TXQVEMFE539A

## STEP 5
- dialogues.ipynb - code to get 4 different continuations of Comment from [a generative dialogue model](https://api.aicloud.sbercloud.ru/public/v2/boltalka/docs#/default/predict_boltalka_predict_post) 

Link to data: https://disk.yandex.ru/d/Yf5ur-en-U58wA

## STEP 6
- work_out_groups.ipynb – ноутбук, в котором работаем с файлами, которые вручную разметила Аня Суханова.

Link to data: 
