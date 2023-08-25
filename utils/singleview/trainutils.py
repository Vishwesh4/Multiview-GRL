from typing import Tuple, Any
from pathlib import Path

import torchmetrics
import torch_geometric

from ..dataloader import BRACS_CellGraph_pos
import trainer

@trainer.Metric.register("dgi")
class Test_Metric(trainer.Metric):
    def get_metrics(self):
        metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(),
             torchmetrics.F1Score(num_classes=self.kwargs["n_classes"], average="weighted")]
        )
        return metricfun

@trainer.Dataset.register("bracs_full_v2")
class Graph_dataset3(trainer.Dataset):
    def get_transforms(self) -> Tuple[Any, Any]:
        return None, None

    def get_loaders(self): 
        trainset = BRACS_CellGraph_pos(root=str(Path(self.path) / "train"), class_name_list=self.kwargs["class_name_list"], feat_folder=self.kwargs["feat_folder"], add_pos=self.kwargs["add_pos"])
        testset = BRACS_CellGraph_pos(root=str(Path(self.path) / "val"), class_name_list=self.kwargs["class_name_list"],feat_folder=self.kwargs["feat_folder"], add_pos=self.kwargs["add_pos"])
        trainloader = torch_geometric.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True)       
        testloader = torch_geometric.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=True)
        print(f"Training size: {len(trainset)}\nValidation size: {len(testset)}")
        return trainset, trainloader, testset, testloader

    def _get_targets(self,data):
        target = []
        for i in range(len(data)):
            target.append(data[i].y)
        return target

@trainer.Dataset.register("bracs_full_v2_test")
class Graph_dataset_test1(trainer.Dataset):
    def get_transforms(self) -> Tuple[Any, Any]:
        return None, None

    def get_loaders(self): 
        trainset = BRACS_CellGraph_pos(root=str(Path(self.path) / "train"), class_name_list=self.kwargs["class_name_list"], feat_folder=self.kwargs["feat_folder"], add_pos=self.kwargs["add_pos"])
        testset = BRACS_CellGraph_pos(root=str(Path(self.path) / "test"), class_name_list=self.kwargs["class_name_list"],feat_folder=self.kwargs["feat_folder"], add_pos=self.kwargs["add_pos"])
        trainloader = torch_geometric.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True)       
        testloader = torch_geometric.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=True)
        print(f"Training size: {len(trainset)}\nTest size: {len(testset)}")
        return trainset, trainloader, testset, testloader

    def _get_targets(self,data):
        target = []
        for i in range(len(data)):
            target.append(data[i].y)
        return target
