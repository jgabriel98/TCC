# TCC

###### Atenção, esse repositório contém um submódulo ([coinmarketcalWebScrapper](https://github.com/jgabriel98/coinmarketcalWebScrapper/blob/master/README.md)), localizado em `./src/data/loaders/coinmarketcalWebScrapper`, caso deseje executar código deste repositório, certifique-se que o submódulo foi clonado também.

![ETH trained - EOS forecasting.png](resultados/generic_model/ETH%20trained%20-%20EOS%20forecasting.png)
![results table](https://github.com/jgabriel98/TCC/assets/37881981/d625569b-93cd-4307-bb52-04e149f376d5)

## Documento pdf do TCC:
 [TCC_JoaoGabriel.pdf](TCC%20-%20escrita/TCC_JoaoGabriel.pdf)

## Resultados e estatísticas
* Comparativo pré-testes - preço "crú" VS variação do preço:
  - estatisticas em: `./src/LSTM_BTC_raw_vs_variation.ipynb`
  - plots (graficos) das predições: `./src/LSTM.ipynb` está mais para o inicio, acaba antes da metade
* Modelos genéricos:
  - Plots dos resultados: `./resultados/generic_model/<moeda baseTreino> trained - <moeda alvoTeste> forecasting.png`
  - Estatísticas dos resultados: `./src/LSTM_GenericModel_<moeda baseTreino>-<moeda alvoTeste>.ipynb`
  
* Features Adicionais:
  - Testes com indicadores técnicos (Bitcoin): `./src/LSTM_technicalFeatures.ipynb`
  - Teste com indicadores técnidos (restante das moedas): `./src/LSTM_technicalFeatures_with_altcoins.ipynb`
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
  
-----------------------------------
### O restante dos arquivos não citados aqui ou não referenciados no arquivo pdf do TCC , podem estar depreciados e não remeterem à capacidade e performance do Modelo.
