| Normalização     | norm_window<br/><sup><sup><sub>parâmetro para o método de normalização</sub></sup></sup> | N  | loss    |   |
|------------------|:-----------:|----|:-------:|---|
| `padrao`         | -           | 90 | 0.00954 |   |
| `nsteps_behind`  | 500         | 90 | 0.00098 |   |
| `nsteps_behind`  | 500         | 60 | 0.00200 |   |
| `nsteps_behind`  | 1000        | 90 | 0.00084 |   |
| `nsteps_arround` | 500         | 90 | 0.00075 |   |
| `nsteps_arround` | 500         | 60 | 0.00087 |   |
| `nsteps_arround` | 1000        | 90 | 0.00067 |   |
| nsteps_arround   | 800         | 90 | 0.00057 |   |




## após correcao no código:


| Normalização     | norm_window<br/><sup><sup><sub>parâmetro para o método de normalização</sub></sup></sup> | N  | loss    |
|------------------|:-----------:|----|:-------:|
| `padrao`         | -           | 90 |  |
| `nsteps_behind`  | 500         | 90 |  |
| `nsteps_behind`  | 500         | 60 | 0.00122 |
| `nsteps_behind`  | 1000        | 90 |  |
| `nsteps_arround` | 500         | 60 | 0.00075 |
| `nsteps_arround` | 500         | 90 | 0.00090 |
| `nsteps_arround` | 700         | 60 | 0.00055 |
| `nsteps_arround` | 800         | 60 | 0.00055 |
| `nsteps_arround` | 800         | 90 | 0.00100 |
| `nsteps_arround` | 1000        | 60 | 0.00070 |
| `nsteps_arround` | 1000        | 90 | 0.00087 |



## após correção 2 no código (10/04/2021):

| Normalização<br/> de preço     | norm_window<br/><sup><sup><sub>parâmetro para o método de normalização</sub></sup></sup> | N  | Bidirecional | text features |loss train   |  loss test  |
|:------------------:|:-----------:|----|:---:|:------:|:-------:|:--------:|
| `padrao`         | -           | 40 | sim |    -   | 0.000320 | 0.000515 |
| `padrao`         | -           | 40 | sim | trends | 0.000188 | 0.000551 | 
| `padrao`         | -           | 40 | sim | tweets | 0.000208 | 0.001083 |
| `padrao`         | -           | 40 | sim |trend+tweet| 0.000139 | 0.000547 |
| `padrao`         | -           | 60 | sim | - | 0.000356 | 0.000486 | 
| `padrao`         | -           | 60 | sim | trends | 0.000224 | 0.000717 | 



obs: nada vs trends -> trend parece ser um pouco melhor: rede compare os gráficos








## DEPRECIADO | SERÁ USADO COMO BLOCO DE ANOTAÇÕES

0,45,LTC,80,"prices,variation",Bidirectional(LSTM(150)) > Dense(50) > Dense(1),0.0002304128138348,0.0002514157094992,-0.3052386939525604,1: 
 - apresentou um inicio bom, e fim ok


1,45,LTC,80,"prices,variation",Bidirectional(LSTM(150)) > Dense(100) > Dense(50) > Dense(1),0.0002360113721806556,0.0002685193612705916,-0.10413817316293716,1
 - apresentou um inicio com bias para queda, e fim muito bom