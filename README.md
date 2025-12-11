# RecursiveMamba implementation

## Build the dataset

We have followed the following repository to build the dataset including ARC AGI 1, ARC AGI 2 and Maze 30x30. 
https://github.com/SamsungSAILMontreal/TinyRecursiveModels

Additionally, we have built our own Maze 100x100 dataset and it can be found in the following hugging face link:
https://huggingface.co/datasets/corl0s/maze-100x100-wilson-hard-1k

We have already setup the respective config files which runs based on the report's specifications. The architecture which will be used is Vision Mamba.

To train the model, run the following command:

1. Maze 30x30 dataset. 
```python
python pretrain.py arch=trm_mamba data_paths="[data/maze-30x30-hard-1k]" evaluators="[]" epochs=1000 eval_interval=100 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_att_maze30x30 ema=True
```

2. Maze 100x100 dataset.
```python
python pretrain.py arch=trm_mamba data_paths="[data/maze-100x100-hard-1k]" evaluators="[]" epochs=1000 eval_interval=100 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_att_maze100x100 ema=True
```

3. ARC-AGI 1 dataset.
```python
python pretrain.py arch=trm_mamba data_paths="[data/arc1concept-aug-1000]" epochs=100 eval_interval=10 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_att_arc1concept_4 ema=True
```

4. ARC-AGI 2 dataset.
```python
python pretrain.py arch=trm_mamba data_paths="[data/arc2concept-aug-1000]" epochs=100 eval_interval=10 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_att_arc2concept_4 ema=True
```


To test our code, edit the config files to include the trained .pth files, required hyperparameters and run:

```python
python run_test.py
```
