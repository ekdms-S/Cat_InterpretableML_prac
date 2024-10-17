# Cat_InterpretableML
data preprocessing, model implementation and GradCAM within interpretable ML framework

## Installation
Run the following command to set up
```
conda update conda
conda create -n Cat_InterpretableML python=3.8
pip install galvani
pip install pandas
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install umap-learn
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Exmaple Use
### Input files generation
To prepare the input of the model, you will need to [download all data](#data) that we provide. Then, please run:
```
python data/LSV_preprocessor.py
python data/feats_preprocessor.py
```
This would not take long time

### Train model
Hyperparameters, such as `--train_system`, `--batch_size`, `--init_lr`, `--epoch` can be set. 
Their default values are set as 'AgNi', 256, 0.00075 and 1000, respectively, now.
```
python train.py
```

### Evaluate trained model
Trained model can be evaluated, specifying training conditions through `--ckpt_filename`, `--train_system`, `--extrap` settings fields. Results in the paper can be reproduced with `--ckpt_filename='ResCBAM_best_checkpoint.pkl'`.
```
python analysis/performance.py
```

### XAI interpretation
After training model, you can obtain Grad-CAM attention map, specifying training and operating conditions. This is similar to the evaluation option above. 
```
python analysis/XAI.py --catalyst='Ag' --voltage=-3.2
```
One may want to interpret averaged Grad-CAM attention map. Then, add `--ensemble` field as below. Randomly initialized models discussed in the paper are in `..\saved\AgNi\ensemble` directory, now. `--ensemble` option autonmatically load those models and average their results.
```
python analysis/XAI.py --catalyst='Ag' --voltage=-3.2 --ensemble
```

## Data
All data used in this research can be downloaded from the link below.

<https://www.kist-cepl.com>
