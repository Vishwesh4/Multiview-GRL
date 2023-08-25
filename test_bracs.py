import random
from copy import deepcopy

import torch
import numpy as np
import torchmetrics
from torchmetrics.classification import MulticlassF1Score

import utils
import trainer

def test(dataset, device, metrics, model, type=None, encodertype="patch"):
        if type=="multiview":
                x_train,y_train,names = get_embeddings_multiview(model=model,dataloader = dataset.trainloader,device=device, encodertype=encodertype)
        else:
                x_train,y_train = get_embeddings(model=model,dataloader = dataset.trainloader,device=device)
        acc = utils.evaluate.train_logistic_classify(x_train, y_train, device)
        print("Training Accuracy: {}".format(acc))
        if type=="multiview":
                x_test,y_test,names = get_embeddings_multiview(model=model,dataloader = dataset.testloader,device=device, encodertype=encodertype)
        else:
                x_test,y_test = get_embeddings(model=model,dataloader = dataset.testloader,device=device)
        preds = utils.evaluate.val_logistic_classify(x_train, y_train, x_test, device)
        metrics(preds.to(device), y_test.to(device))
        metrics.compute()

        print("Test F1 Score: {}".format(metrics.results["test_F1Score"]))
        print("Test Accuracy: {}".format(metrics.results["test_Accuracy"]))
        print("Test Confusion Matrix: {}".format(metrics.results["test_ConfusionMatrix"]))
        temp_mat = metrics.results["test_ConfusionMatrix"]

        print("Multiclass F1 score: {}".format(metrics.results["test_MulticlassF1Score"]))
        return metrics.results["test_F1Score"], metrics.results["test_MulticlassF1Score"], np.sum(temp_mat,axis=1)

def get_embeddings(model,dataloader,device):
    #Perform again to get the embedding for the epoch
    model.eval()
    x_data = []
    y_data = []
    with torch.no_grad():
        for data in dataloader:
            # store embedding of training data
            x,edge_index,batch,y = data.x, data.edge_index, data.batch, data.y
            if x is not None:
                x = x.to(device)
            edge_index, batch = edge_index.to(device), batch.to(device)
            graph_embeddings , _, batch = model.get_encoder(x,edge_index,batch)
            x_data.append(graph_embeddings.cpu())
            y_data.append(y)
    x_data = torch.concat(x_data)
    y_data = torch.ravel(torch.concat(y_data))
    return x_data, y_data

def get_embeddings_multiview(model,dataloader,device,encodertype):
    #Perform again to get the embedding for the epoch
    model.eval()
    x_data = []
    y_data = []
    names = []
    with torch.no_grad():
        for data in dataloader:
            # store embedding of training data
            data_cell, data_patch = data
            graph_embeddings = model.get_representation(data_cell, data_patch, device,type=encodertype)
            x_data.append(graph_embeddings.cpu())
            y_data.append(data_cell.y)
            names.extend([img.stem for img in data_cell.img_path])
    x_data = torch.concat(x_data)
    y_data = torch.ravel(torch.concat(y_data))
    return x_data, y_data, names

def setup_seed(random_seed):
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)

def get_test_metrics(model_dict, dataset, metrics, model_files, device, viewtype, encodertype):
  f1s = []
  f1slist = []
  for i,files in enumerate(model_files):
    setup_seed(i)
    model_dict_copy= deepcopy(model_dict)
    print("-"*50)
    print(files)
    model = trainer.Model.create(**model_dict_copy)
    model.load_model_weights(model_path=files,device=device)
    model = model.to(device)
    model.eval()
    f1score, f1_list, n_samples = test(dataset,device,metrics,model,type=viewtype,encodertype=encodertype)
    f1s.append(f1score)
    f1slist.append(f1_list)
  print("For latex")
  latstr = ""
  means = 100*np.mean(f1slist,axis=0)
  stds = 100*np.std(f1slist,axis=0)
  for i in range(len(means)):
    latstr+=" ${:.2f}\pm{:.2f}$ &".format(round(means[i],2),round(stds[i],2))
  latstr+=" ${:.2f}\pm{:.2f}$".format(round(np.mean(f1s)*100,2),round(np.std(f1s)*100,2))
  print(latstr)

@trainer.Metric.register("bracs_test1")
class Test_Metric2(trainer.Metric):
    def get_metrics(self):
        metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(),
             torchmetrics.F1Score(num_classes=self.kwargs["n_classes"], average="weighted"),
             torchmetrics.ConfusionMatrix(num_classes=self.kwargs["n_classes"]),
             MulticlassF1Score(num_classes=self.kwargs["n_classes"], average=None)]
        )
        return metricfun
    
#### Define all the required variables
#change from here
cg_dataclass_dict={
  "subclass_name": "bracs_full_v2_test",
  "path": "./BRACS_dataset_prev",
  "feat_folder": "graph_obj_simclr_v2",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 12,
  "test_batch_size": 12}

pg_dataclass_dict={
  "subclass_name": "bracs_full_v2_test",
  "path": "./BRACS_dataset_prev",
  "feat_folder": "patch_graph_custom_simclr",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 12,
  "test_batch_size": 12}

mv_dataclass_dict={
  "subclass_name": "multiview_data_test",
  "path": "./BRACS_dataset_prev",
  "feat_folder_cell": "graph_obj_simclr_v2",
  "feat_folder_patch": "patch_graph_custom_simclr",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 32,
  "test_batch_size": 32}

cg_dataclass_dict_val={
  "subclass_name": "bracs_full_v2",
  "path": "./BRACS_dataset_prev",
  "feat_folder": "graph_obj_simclr_v2",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 12,
  "test_batch_size": 12}

pg_dataclass_dict_val={
  "subclass_name": "bracs_full_v2",
  "path": "./BRACS_dataset_prev",
  "feat_folder": "patch_graph_custom_simclr",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 12,
  "test_batch_size": 12}


mv_dataclass_dict_val={
  "subclass_name": "multiview_data",
  "path": "./BRACS_dataset_prev",
  "feat_folder_cell": "graph_obj_simclr_v2",
  "feat_folder_patch": "patch_graph_custom_simclr",
  "add_pos": False,
  "class_name_list": ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA" , "5_DCIS", "6_IC"], #change for prev dataset
  "train_batch_size": 32,
  "test_batch_size": 32}


pg_model_dict={
  "subclass_name": "bracs_disc",
  "model_name": "GNN_ASAP_bn",
  "in_channel": 512,
  "hidden_channel": 522,
  "out_channel": 256,
  "num_gc_layers": 4,
  "batch_norm": True,
  "pooling": False
}

cg_model_dict={
  "subclass_name": "bracs_disc",
  "model_name": "GNN_ASAP_bn",
  "in_channel": 518,
  "hidden_channel": 256,
  "out_channel": 256,
  "num_gc_layers": 4,
  "batch_norm": True,
  "pooling": False
}

mv_model_dict={
  "subclass_name": "multiview_contrast_sample_v2",
  "out_channel": 256,
  "num_samples": 500,
  "sample_proc": "bottom",
  "proc_percent": 0.5,
  "cellgraph_params":{
    "cellgraph_name": "GNN_ASAP_bn",
    "out_channel": 256,
    "in_channel": 518,
    "hidden_channel": 256,
    "num_gc_layers": 4,
    "batch_norm": True,
    "pooling": False},
  "patchgraph_params":{
    "patchgraph_name": "GNN_ASAP_bn",
    "out_channel": 256,
    "in_channel": 512,
    "hidden_channel": 522,
    "num_gc_layers": 4,
    "batch_norm": True,
    "pooling": False}
}

logger_dict={
  "use_wandb": False, 
  "project_name": "BRACS",
  "run_name": "test metrics",
  "notes": "testing pipeline"
}


pg_dicts = (pg_dataclass_dict,pg_dataclass_dict_val,pg_model_dict)
cg_dicts = (cg_dataclass_dict,cg_dataclass_dict_val,cg_model_dict)
mv_dicts = (mv_dataclass_dict,mv_dataclass_dict_val,mv_model_dict)

num_classes = 7
device = torch.device("cuda:0")
# Build classes
logger = trainer.Logger(**logger_dict, configs=logger_dict)
metrics = trainer.Metric.create(subclass_name= "bracs_test1", n_classes = num_classes, logger=logger, device=device)
metrics.mode="test" #only affects name of torchmetrics output dictionary

####################################################################################################################################################

dataclass_dict, valclass_dict, model_dict = mv_dicts
dataset = trainer.Dataset.create(**dataclass_dict)
mv_files = ["./saved_models/multiview/Checkpoint_mv_seed_0.pt"]
get_test_metrics(model_dict, dataset, metrics, mv_files, device, viewtype="multiview", encodertype="patch")
get_test_metrics(model_dict, dataset, metrics, mv_files, device, viewtype="multiview", encodertype="cell")

dataclass_dict, valclass_dict, model_dict = pg_dicts
dataset = trainer.Dataset.create(**dataclass_dict)
pg_files = ["./saved_models/infograph_patch/Checkpoint_pg_seed_0.pt"]
get_test_metrics(model_dict, dataset, metrics, pg_files, device, viewtype="singleview", encodertype="patch")

dataclass_dict, valclass_dict, model_dict = cg_dicts
dataset = trainer.Dataset.create(**dataclass_dict)
cg_files = ["./saved_models/infograph_cell/Checkpoint_cg_seed_0.pt"]
get_test_metrics(model_dict, dataset, metrics, cg_files, device, viewtype="singleview", encodertype="cell")
