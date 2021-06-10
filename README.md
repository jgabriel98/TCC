# TCC

###### Atenção, esse repositório contém um submódulo ([coinmarketcalWebScrapper](https://github.com/jgabriel98/coinmarketcalWebScrapper/blob/master/README.md)), localizado em `./src/data/loaders/coinmarketcalWebScrapper`, caso deseje executar código deste repositório, certifique-se que o submódulo foi clonado também.

## Resultados e estatísticas
* Modelos genéricos:
  - Plots dos resultados: `./resultados/generic_model/<moeda baseTreino> trained - <moeda alvoTeste> forecasting.png`
  - Estatísticas dos resultados: `./src/LSTM_GenericModel_<moeda baseTreino>-<moeda alvoTeste>.ipynb`
  
* Features Adicionais:
  - Testes com indicadores técnicos: `./src/LSTM_technicalFeatures.ipynb`
  - Testes com dados sociais: `./src/LSTM.ipynb` começe olhando pelo final, pois tem muita coisa antes

## Dados
 - dados do kaggle: `./data/kaggle - Cryptocurrency Historical Prices/coin_<moeda>.csv`
 - dados sociais: `./data/social_data<moeda>.csv` . Esses arquivos são gerados quando a função `load_data()` (encontrada em `src/data/loades/utils.py`) 
 é chamada e o arquivo de dados sociais desta moeda não existe (é preciso ter o selenium instalado e funcionando nesse caso, e algumas outras dependências que não lembro mais).
 
 ## código fonte
 o código fonte se encontra na pasta `./src`
  - `./src/data` contém arquivos relacionados a obtenção e tratamento de dados.<br/>
    \`→ para tratamento de dados `utils.py` é o arquivo mais relevante.
  - `./src/data/loaders` contém as classes para extração de dados sociais.
  - `./src/metrics/custom.py` contém as funções das métricas, as relevantes são: `mean_squared_error()`, `custom_movement_accuracy()` e `above_or_below_zero_accuracy()`
  
