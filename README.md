# Dataset-Cartography
**Dataset Cartography for human alignment** 
@LDI Sogang. 2024

## Requirements
```
conda env create -f environment.yml
conda activate dc
pip install -r requirements.txt
```

## Codebase Structure
```
Dataset-Cartography/
├── data-map/                   # Generating Data maps 
│   ├── results/
│   └── src/ 
├── train-eval/                 # for Train and Evaluation                 
│   ├── LLaMA-Factory/          # for DPO Train
│   ├── SimPO/                  # for SimPO Train
│   ├── src/                    # Evaluation Code 
│   └── eval/                   
│       ├── alpaca_eval 
│       ├── instruct-eval           
│       └── FastChat          
├── README.md                  
├── requirements.txt            
└── environment.yml         
```

## How to Run data-map 
### 1. GPT Inference 
Choose model to use and process openAI API Inference <br>
config : ./src/config.yaml
```
python gpt-inference.py
```

### 2. Similarity Score Calculation
calculating similarity of proxy GPT response and other model responses using various Sentence  Transformer models <br>
config : ./src/scores/sentence-transformer-config.yaml <br>
available models : ST(Sentence Transformer), NV
```
python sentence-transformer.py
```

### 3. Variance Calculation
calculating variance between responses <br>
config : ./src/scores/variance-config.yaml <br>
available methods : proxy-response, response
```
python variance.py
```

### 4. Correlation Calculation 
calculating correlation of human preference and proxy-response similarity score using various methods <br>
config : ./src/correlations/correlation-config.yaml <br>
available methods : cosine, cosine-minmax, pearson, pearson-minmax, kendall
```
python correlation-calculation.py
```

### 5. Visualization
./src/visualization

