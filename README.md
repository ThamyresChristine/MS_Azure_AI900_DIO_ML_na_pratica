# Modelo de Previsão através do Azure Machine Learning
_Projeto realizado para a Certificação IA900 oferecida pela Microsoft em parceria com a DIO - Digital Innovation One_

---
## Escolha do Conjunto de Dados e do Modelo Preditivo
---
O conjunto de dados escolhido foi um Banco de Dados de pacientes com possibilidade de câncer de pulmão [você pode acessá-lo aqui](https://data.world/cancerdatahp/lung-cancer-data). O conjunto apresenta as seguintes variáveis:
- Age (Idade)
- Gender (Gênero)
- Air Pollution (Poluição do Ar)
- Alcohol use (Uso de Álcool)
- Dust Allergy (Alergia à poeira)
- OccuPational Hazards (Riscos Ocupacionais)
- Genetic Risk (Risco Genético)
- Chronic Lung Disease (Doença pulmonar crônica)
- Balanced Diet (Dieta Balanceada)
- Obesity (Obesidade)
- Smoking (Fumante)
- Passive Smoker (Fumante Passivo)
- Chest Pain (Dor no Peito)
- Coughing of Blood (Tosse com Sangue)
- Fatigue (Fatiga)
- Weight Loss (Perda de Peso)
- Shortness of Breath (Dificuldade em respirar)
- Wheezing (Chiado)
- Swallowing Difficulty (Dificuldade de deglutição)
- Clubbing of Finger Nails (Baqueteamento das unhas)
- Frequent Cold (Frio Frequente)
- Dry Cough (Tosse Seca)
- Snoring (Ronco)
- Level(Nível)

Analisando os dados, percebe-se que todas as variáveis são quantitativas (com valor mínimo 1 e máximo 9), com exceção apenas uma variável quantitativa - _Level_:

```
import pandas as pd
f = open("C:\\Users\\thamy\\Área de Trabalho\\VisualStudioCode\\Python_Bible_Study\\Data\\cancer_patient_data_sets.csv")
data = pd.read_csv(f, sep=";")
data.head()
```
```
|    | ï»¿Patient Id   |   Age |   Gender |   Air Pollution |   Alcohol use |   Dust Allergy |   OccuPational Hazards |   Genetic Risk |   chronic Lung Disease |   Balanced Diet |   Obesity |   Smoking |   Passive Smoker |   Chest Pain |   Coughing of Blood |   Fatigue |   Weight Loss |   Shortness of Breath |   Wheezing |   Swallowing Difficulty |   Clubbing of Finger Nails |   Frequent Cold |   Dry Cough |   Snoring | Level   |
|---:|:----------------|------:|---------:|----------------:|--------------:|---------------:|-----------------------:|---------------:|-----------------------:|----------------:|----------:|----------:|-----------------:|-------------:|--------------------:|----------:|--------------:|----------------------:|-----------:|------------------------:|---------------------------:|----------------:|------------:|----------:|:--------|
|  0 | P1              |    33 |        1 |               2 |             4 |              5 |                      4 |              3 |                      2 |               2 |         4 |         3 |                2 |            2 |                   4 |         3 |             4 |                     2 |          2 |                       3 |                          1 |               2 |           3 |         4 | Low     |
|  1 | P10             |    17 |        1 |               3 |             1 |              5 |                      3 |              4 |                      2 |               2 |         2 |         2 |                4 |            2 |                   3 |         1 |             3 |                     7 |          8 |                       6 |                          2 |               1 |           7 |         2 | Medium  |
|  2 | P100            |    35 |        1 |               4 |             5 |              6 |                      5 |              5 |                      4 |               6 |         7 |         2 |                3 |            4 |                   8 |         8 |             7 |                     9 |          2 |                       1 |                          4 |               6 |           7 |         2 | High    |
|  3 | P1000           |    37 |        1 |               7 |             7 |              7 |                      7 |              6 |                      7 |               7 |         7 |         7 |                7 |            7 |                   8 |         4 |             2 |                     3 |          1 |                       4 |                          5 |               6 |           7 |         5 | High    |
|  4 | P101            |    46 |        1 |               6 |             8 |              7 |                      7 |              7 |                      6 |               7 |         7 |         8 |                7 |            7 |                   9 |         3 |             2 |                     4 |          1 |                       4 |                          2 |               4 |           2 |         3 | High    |
```
```
data.dtypes

ï»¿Patient Id               object
Age                          int64
Gender                       int64
Air Pollution                int64
Alcohol use                  int64
Dust Allergy                 int64
OccuPational Hazards         int64
Genetic Risk                 int64
chronic Lung Disease         int64
Balanced Diet                int64
Obesity                      int64
Smoking                      int64
Passive Smoker               int64
Chest Pain                   int64
Coughing of Blood            int64
Fatigue                      int64
Weight Loss                  int64
Shortness of Breath          int64
Wheezing                     int64
Swallowing Difficulty        int64
Clubbing of Finger Nails     int64
Frequent Cold                int64
Dry Cough                    int64
Snoring                      int64
Level                       object
dtype: object
```
O conjunto possui 25 variáveis, com dados de 1000 pacientes>
```
print(data.shape)
print(data.size)

(1000, 25)
25000
```
A transformação da variável _Level_ em quantitativa, equivalendo Low = 1, Medium = 2 e High = 3:
```
data.loc[ data['Level'] == 'Low', 'Level'] = 1
data.loc[ data['Level'] == 'Medium', 'Level'] = 2
data.loc[ data['Level'] == 'High', 'Level'] = 3
data
```
Algumas métricas estatísticas sobre as variáveis do conjunto:
- contagem
- média
- desvio padrão
- valor mínimo
- 1ª quartil
- mediana
- 3 quartil
- valor máximo
```
|       |       Age |      Gender |   Air Pollution |   Alcohol use |   Dust Allergy |   OccuPational Hazards |   Genetic Risk |   chronic Lung Disease |   Balanced Diet |    Obesity |   Smoking |   Passive Smoker |   Chest Pain |   Coughing of Blood |    Fatigue |   Weight Loss |   Shortness of Breath |   Wheezing |   Swallowing Difficulty |   Clubbing of Finger Nails |   Frequent Cold |   Dry Cough |    Snoring |
|:------|----------:|------------:|----------------:|--------------:|---------------:|-----------------------:|---------------:|-----------------------:|----------------:|-----------:|----------:|-----------------:|-------------:|--------------------:|-----------:|--------------:|----------------------:|-----------:|------------------------:|---------------------------:|----------------:|------------:|-----------:|
| count | 1000      | 1000        |       1000      |    1000       |     1000       |             1000       |       1000     |             1000       |      1000       | 1000       | 1000      |       1000       |   1000       |          1000       | 1000       |    1000       |            1000       | 1000       |              1000       |                 1000       |       1000      |  1000       | 1000       |
| mean  |   37.174  |    1.402    |          3.84   |       4.563   |        5.165   |                4.84    |          4.58  |                4.38    |         4.491   |    4.465   |    3.948  |          4.195   |      4.438   |             4.859   |    3.856   |       3.855   |               4.24    |    3.777   |                 3.746   |                    3.923   |          3.536  |     3.853   |    2.926   |
| std   |   12.0055 |    0.490547 |          2.0304 |       2.62048 |        1.98083 |                2.10781 |          2.127 |                1.84852 |         2.13553 |    2.12492 |    2.4959 |          2.31178 |      2.28021 |             2.42796 |    2.24462 |       2.20655 |               2.28509 |    2.04192 |                 2.27038 |                    2.38805 |          1.8325 |     2.03901 |    1.47469 |
| min   |   14      |    1        |          1      |       1       |        1       |                1       |          1     |                1       |         1       |    1       |    1      |          1       |      1       |             1       |    1       |       1       |               1       |    1       |                 1       |                    1       |          1      |     1       |    1       |
| 25%   |   27.75   |    1        |          2      |       2       |        4       |                3       |          2     |                3       |         2       |    3       |    2      |          2       |      2       |             3       |    2       |       2       |               2       |    2       |                 2       |                    2       |          2      |     2       |    2       |
| 50%   |   36      |    1        |          3      |       5       |        6       |                5       |          5     |                4       |         4       |    4       |    3      |          4       |      4       |             4       |    3       |       3       |               4       |    4       |                 4       |                    4       |          3      |     4       |    3       |
| 75%   |   45      |    2        |          6      |       7       |        7       |                7       |          7     |                6       |         7       |    7       |    7      |          7       |      7       |             7       |    5       |       6       |               6       |    5       |                 5       |                    5       |          5      |     6       |    4       |
| max   |   73      |    2        |          8      |       8       |        8       |                8       |          7     |                7       |         7       |    7       |    8      |          8       |      9       |             9       |    9       |       8       |               9       |    8       |                 8       |                    9       |          7      |     7       |    7       |
```







Como agora o conjunto de dados apresenta somente variáveis quantitativas, ele está pronto para ser aplicado um modelo de Machine Learning e o escolhido foi a Regressão Linear.

---
## Montando o modelo de Machine Learning Automatizado no Azure
---
### Criando o Laboratório de ML

---
## Configurando o modelo e os Pontos de Extremidade
---
```
{
  "Inputs": {
    "data": [
      {
        "Patient Id": "example_value",
        "Age": 0,
        "Gender": 0,
        "Air Pollution": 0,
        "Alcohol use": 0,
        "Dust Allergy": 0,
        "OccuPational Hazards": 0,
        "Genetic Risk": 0,
        "chronic Lung Disease": 0,
        "Balanced Diet": 0,
        "Obesity": 0,
        "Smoking": 0,
        "Passive Smoker": 0,
        "Chest Pain": 0,
        "Coughing of Blood": 0,
        "Fatigue": 0,
        "Weight Loss": 0,
        "Shortness of Breath": 0,
        "Wheezing": 0,
        "Swal1ing Difficulty": 0,
        "Clubbing of Finger Nails": 0,
        "Frequent Cold": 0,
        "Dry Cough": 0,
        "Snoring": 0
      }
```

---
Resultado do Teste:
```
{1 item
"Results":[1 item
0:float0.9840139409649559
]
}
