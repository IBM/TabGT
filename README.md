# Tabular Graph Transformer (TabGT)

This repository contains models and scripts for representation learning on large tabular data where each row represents an edge relationship. 

For example,

<div align="center">

| Source | Destination | Column 1 | Column 2 | ... |
|--------|-------------|----------|----------|-----|
| A | B | category 1 | 100.0 | ... |

</div>


The Tabular Graph Transformer (TabGT) combines tabular transformer and graph neural network (GNN) layers by fusing node embeddings from both representations. In this project, [TURL](https://arxiv.org/abs/2006.14806) layers are used as the tabular transformer layer, which acts on the tabular data through row-column attention, while [GIN](https://arxiv.org/abs/1810.00826) is used as the GNN aggregation layer.


## :gear: Setup

Install the conda environment:
```
conda env create -f env.yml
```

## :bulb: Usage

Activate the environment:
```
conda activate tabgt
```

As an example dataset, we download the transactions for Anti Money Laundering (AML) dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data). After downloading it, you first need to perform a pre-processing step:
```
python prepare_AML_transactions.py -i ./HI-Small_Trans.csv -o ./HI-Small_Trans-preprocessed.csv
```

The preprocessed table is saved under `HI-Small_Trans-preprocessed.csv`.

You can then train TabGT on this dataset with the following command:
```
python train.py --include_gnn_node_feats --data_config_path ./data/config/AML_transactions.json --edge_data_path ./HI-Small_Trans-preprocessed.csv --run_name AML
```

The trained model will be saved under a newly created `runs` folder.

## :rocket: Train your own dataset

Any table with categorical and continuous values where each row represents an edge relationship can be used as an input dataset.

Your dataset needs to define 3 files:

* `edge.csv`: contains the edge relationships in tabular format. It must include a source and destination column with unique node IDs. All cell values in each column must either be `nan`, categorical or continuous.
* `node.csv` (optional): contains node features in tabular format. It must include a column with the node ID. All cell values in each column must either be `nan`, categorical or continuous.
* `config.json`: configuration file which specifies which columns in the tables are categorical, continuous, or node IDs. See `data/config/example.json` as an example.

Then, you can train TabGT as follows:
```
python train.py --include_gnn_node_feats --data_config_path <path to config.json> --edge_data_path <path to edge.csv> --node_data_path <path to node.csv> --run_name MyRun
```
In case you do not have a node feature table `node.csv`, you can remove the `--node_data_path` argument.

For more information on modifying hyperparameters and other training configuration run `python train.py --help`.

> You can decrease the GPU memory usage by decreasing the batch size with the `--batch_size` argument

## :balance_scale: License

Apache License Version 2.0, January 2004