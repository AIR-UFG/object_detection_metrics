# Repositório de Avaliação de Métricas e Conversão de JSON

Este repositório contém scripts em Python para avaliar métricas de previsão e converter dados de ground truth e resultados em um formato padronizado JSON para avaliação de modelos de detecção. A organização do repositório facilita a avaliação da precisão de previsões com base em dados de ground truth.

---

## Estrutura

```
├── jsons-conversor
│   ├── GT-json.ipynb       # Notebook para converter dados JSON de ground truth para o formato do nuscenes.
│   └── results-json.ipynb  # Notebook para converter dados JSON de resultados para o formato do nuscenes.
└── metrics_calculator.py
```
---

## Requisitos

Este repositório requer os seguintes pacotes:

- `numpy`
- `scipy`
- `nuscenes-devkit` (para métricas NuScenes)

## Guia de Uso

### 1. Converter Dados JSON

Os notebooks **gt-json.ipynb** e **results-json.ipynb** convertem arquivos JSON de ground truth e de resultados em um formato padronizado utilizado pelo nuscenes.

Ambos os scripts têm pastas padrão definidos para os arquivos de origem e destino, que podem ser modificados dentro dos notebooks.

#### gt-json.ipynb
Converte dados de ground truth em arquivos JSON dentro da pasta `SOURCE_DIR` e salva os arquivos JSON convertidos em `TARGET_DIR`.

#### results-json.ipynb
Converte dados de resultados aplicando um threshold especificado (padrão: 0.5). Os arquivos JSON em `SOURCE_DIR` são processados e salvos em `TARGET_DIR`.

### 2. Avaliação de Métricas

O script **metrics_calculator.py** avalia previsões do modelo calculando métricas, incluindo o mean Average Precision (mAP) para cada classe.

**Parâmetros:**
- `--preds_dir`: Pasta contendo os arquivos de predição.
- `--gt_dir`: Pasta contendo os arquivos de ground truth.
- `--output_dir`: Pasta onde os resultados serão salvos.

**Exemplo de Uso:**

```bash
python metrics_calculator.py --preds_dir caminho/para/predicoes --gt_dir caminho/para/ground_truth --output_dir caminho/para/saida
```

Este script avalia cada arquivo em `preds_dir` e `gt_dir` e salva os resultados em `output_dir` com o mesmo nome de arquivo dos arquivos de entrada, mas com extensão `.txt`.

---

## Workflow

1. **Converter JSONs de Ground Truth**:
   - Execute `gt-json.ipynb` para converter dados brutos de ground truth para o formato de destino.
   
2. **Converter JSONs de Resultados**:
   - Execute `results-json.ipynb` para converter dados de previsão para o formato JSON padronizado, aplicando um threshold.

3. **Avaliar Previsões**:
   - Execute `metrics_calculator.py` para calcular métricas de avaliação e salvar os resultados.

---

# Métricas
 Texto extraído do [site do NuScenes](https://www.nuscenes.org/object-detection) e traduzido.

### Métrica de Average Precision  
**Mean Average Precision (mAP)**: Usamos a métrica de Average Precision bem conhecida, mas definimos uma correspondência considerando a distância do centro 2D no plano do solo em vez de afinidades baseadas em interseção sobre união. Especificamente, correspondemos previsões com objetos do ground truth que têm a menor distância entre centros, até um certo limite. Para um dado limiar de correspondência, calculamos a average precision (AP) integrando a curva de recall vs. precision para recalls e precisions > 0.1. Finalmente, fazemos a média em diferentes limiares de correspondência de {0.5, 1, 2, 4} metros e calculamos a média entre as classes.

### Métricas de True Positive  
Aqui definimos métricas para um conjunto de true positives (TP) que medem erros de translação / escala / orientação / velocidade e atributo. Todas as métricas de TP são calculadas usando um limiar de 2 metros de distância do centro durante a correspondência, e todas são projetadas para serem escalares positivos.

A correspondência e pontuação ocorrem independentemente por classe, e cada métrica é a média da média cumulativa em cada nível de recall alcançado acima de 10%. Se 10% de recall não for alcançado para uma classe específica, todos os erros de TP para essa classe são definidos como 1. Definimos os seguintes erros de TP:

- **Average Translation Error (ATE)**: Distância euclidiana do centro em 2D em metros.
- **Average Scale Error (ASE)**: Calculado como 1 - IOU após o alinhamento de centros e orientação.
- **Average Orientation Error (AOE)**: Menor diferença de ângulo de yaw entre a previsão e o ground-truth em radianos. O erro de orientação é avaliado em 360 graus para todas as classes, exceto para barreiras, onde é avaliado apenas em 180 graus. Erros de orientação para cones são ignorados.
- **Average Velocity Error (AVE)**: Erro absoluto de velocidade em m/s. O erro de velocidade para barreiras e cones é ignorado.
- **Average Attribute Error (AAE)**: Calculado como 1 - acc, onde acc é a precisão da classificação de atributos. O erro de atributo para barreiras e cones é ignorado.

Todos os erros são >= 0, mas observe que para erros de translação e velocidade, os erros são ilimitados e podem ser qualquer valor positivo.

As métricas de TP são definidas por classe, e então fazemos uma média entre as classes para calcular mATE, mASE, mAOE, mAVE e mAAE.

nuScenes detection score  
nuScenes detection score (NDS): Consolidamos as métricas acima calculando uma soma ponderada: mAP, mATE, mASE, mAOE, mAVE e mAAE. Como primeiro passo, convertimos os erros de TP em pontuações de TP como TP_score = max(1 - TP_error, 0.0). Em seguida, atribuímos um peso de 5 para mAP e 1 para cada uma das 5 pontuações de TP e calculamos a soma normalizada.