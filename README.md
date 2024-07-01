# Self Supervised Multi-View Graph Representation Learning in Digital Pathology

This repository contains the code for **Self Supervised Multi-View Graph Representation Learning in Digital Pathology** submitted to [GRAIL 2023](https://grail-miccai.github.io/). Please refer to the [paper](https://link.springer.com/chapter/10.1007/978-3-031-55088-1_7) for more details.

## Description
The shortage of annotated data in digital pathology presents a significant challenge for training GNNs. Inspired by pathologists who take multiple views of a histology slide under a microscope for exhaustive analysis, this project performs graph representation learning using self-supervision. The methodology leverages multiple graph views constructed from a given histology image to capture diverse information by maximizing mutual information across nodes and graph representations of different graph views, resulting in a comprehensive graph representation. The trained graph encoder weights for Feature Extractor (SimCLR),Infograph Cell, Infograph Patch and MultiView Graph is shared at this [link](https://drive.google.com/drive/folders/1myOGN-dKp8oG2460GjR0QzXKV6hcbhz-?usp=sharing).

<img src="https://github.com/Vishwesh4/Multiview-GRL/blob/master/images/methodology_mv.png" align="center" width="880" ><figcaption>Fig.1 - Methodology overview</figcaption></a>

<img src="https://github.com/Vishwesh4/Multiview-GRL/blob/master/images/bracs_retrieval.png" align="center" width="880" ><figcaption>Fig.2 - Retrieving image patches from the BRACS testset based on MultiView Patch graph representation of query images. Among the retrieved
images, the incorrect class is marked in red, and the correct class in green. </figcaption></a>
## Getting Started

### Dependencies

```
opencv
dgl (only for farthest sampling method used while building cell graph)
torch-geometric
trainer - https://github.com/Vishwesh4/TrainerCode
torchmetrics
wandb
networkx
sklearn
torchvision
```
### Dataset
The experiments was performed on BRACS dataset previous version downloaded from the [website](https://www.bracs.icar.cnr.it/).

### Preprocessing
For performing training on histology images using the proposed methodology, the images were converted into cell graph and patch graph.

For generating feature embedding for the nodes for both the graphs, Resnet34 was trained on BRACS training dataset using SimCLR. The model weight is shared in the same drive link.

#### Cell graph
For generating cell graph, the cells were identified using hovernet using panuke pretrained model [link](https://github.com/vqdang/hover_net). The generated `.mat` files and histology images were used for generating cell graph using
```
python utils/build_cellgraph_full.py
```
#### Patch graph
For generating patch graph use
```
python utils/build_patchgraph_full.py
```
### Training using infomax
#### Single View
The infograph model for both cell graph and patch graph can be trained using infomax using
```
python train_infograph.py -c ./configs/infograph_[cell/patch].yml -s [SEED]
```
The training hyperparameters can be directly modifed from the config files present in the `config` folder

#### Multi View
The cell and patch graph encoders can be trained using same-view and cross-view mutual information maximization using 
```
python train_multiview.py -c ./configs/multiview.yml -s [SEED]
``` 
### Deploying
The trained weights can be used for transfer learning like shown in `./test_bracs.py`
## Authors
- Vishwesh Ramanathan ([@Vishwesh4](https://github.com/Vishwesh4))
## Contact
If you want to contact, you can reach the authors by raising an issue or
 email at vishwesh.ramanathan@mail.utoronto.ca

## Acknowledgments
- Code was inspired from Infograph [repository](https://github.com/sunfanyunn/InfoGraph) and Pytorch Geometric [repository](https://github.com/pyg-team/pytorch_geometric/tree/master)

## Cite
```
@inproceedings{ramanathan2023self,
  title={Self Supervised Multi-view Graph Representation Learning in Digital Pathology},
  author={Ramanathan, Vishwesh and Martel, Anne L},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={74--84},
  year={2023},
  organization={Springer}
}
```
