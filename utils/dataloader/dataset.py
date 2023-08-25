import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data

class BRACS_CellGraph_pos(Dataset):
    """
    In memory dataset for BRACS dataset. Loads the processed graph data files
    """
    # CLASS_NAMES = ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]
    # CLASS_NAMES = ["2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

    def __init__(self,root:str,class_name_list:list,feat_folder:str="graph_obj", add_pos:bool=False, transform=None):
        super().__init__()
        self.root = Path(root)
        self.class_names = class_name_list
        self.feat_folder = feat_folder
        self.add_pos = add_pos
        self.transform = transform
        #Load all the graphs
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads the data present in a specific format
        """
        data = []
        self.all_files = []
        for i,classnames in enumerate(self.class_names):
            graph_path = self.root / Path(classnames) / self.feat_folder
            file_list = list(graph_path.glob("*.pt"))
            for idx,files in tqdm(enumerate(file_list), desc=f"Loading files from {classnames}"):
                temp_data = torch.load(files,map_location=torch.device("cpu"))
                temp_data.coords = temp_data.x[:,-2:].detach().numpy()
                if not self.add_pos:
                    #remove last two values which has coordinate information
                    temp_data.x = temp_data.x[:,:-2]
                temp_data.x = temp_data.x.to(dtype=torch.float32)
                temp_data.y = i
                temp_data.img_path = files.parent.parent / "overlay" / (files.stem+".png")
                temp_data.idx = idx
                data.append(temp_data)
                self.all_files.append(files)
        return data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int)->Data:
        if self.transform is not None:
            return self.transform(self.data[index])
        return self.data[index]

class BRACS_CellGraph_multi(Dataset):
    """
    In memory dataset for BRACS dataset. Loads the processed graph data files
    """
    # CLASS_NAMES = ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]
    # CLASS_NAMES = ["2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

    def __init__(self,root:str, class_name_list:list, feat_folder_cell:str, feat_folder_patch:str, add_pos:bool=False):
        super().__init__()
        self.root = Path(root)
        self.class_names = class_name_list
        self.feat_folder_cell = feat_folder_cell
        self.feat_folder_patch = feat_folder_patch
        self.add_pos = add_pos
        #Load all the graphs
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads the data present in a specific format
        """
        data_patch = []
        data_cell = []
        self.all_files = []
        for i,classnames in enumerate(self.class_names):
            graph_path_patch = self.root / Path(classnames) / self.feat_folder_patch
            graph_patch_cell = self.root / Path(classnames) / self.feat_folder_cell
            file_list = list(graph_path_patch.glob("*.pt"))
            for idx,files in tqdm(enumerate(file_list), desc=f"Loading files from {classnames}"):
                #Patch_graph
                temp_data = torch.load(files,map_location=torch.device("cpu"))
                temp_data.coords = temp_data.x[:,-2:].detach().numpy()
                if not self.add_pos:
                    #remove last two values which has coordinate information
                    temp_data.x = temp_data.x[:,:-2]
                temp_data.x = temp_data.x.to(dtype=torch.float32)
                temp_data.y = i
                temp_data.img_path = files.parent.parent / "overlay" / (files.stem+".png")
                temp_data.idx = idx
                data_patch.append(temp_data)

                #Corresponding Cell_graph
                file_cell = graph_patch_cell / files.name
                temp_data = torch.load(file_cell,map_location=torch.device("cpu"))
                temp_data.coords = temp_data.x[:,-2:].detach().numpy()
                if not self.add_pos:
                    #remove last two values which has coordinate information
                    temp_data.x = temp_data.x[:,:-2]
                temp_data.x = temp_data.x.to(dtype=torch.float32)
                temp_data.y = i
                temp_data.img_path = file_cell.parent.parent / "overlay" / (files.stem+".png")
                temp_data.idx = idx
                data_cell.append(temp_data)

                self.all_files.append(files)
        return data_cell, data_patch
    
    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, index:int)->Data:
        return self.data[0][index], self.data[1][index]   
