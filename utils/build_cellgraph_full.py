#
# Description: This script is about making cell graphs using output mat from hovernet model, Here I have used KNN neighbour with threshold. The points has been sampled
# using farthest point sampler + random as suggested in a paper. Also build feature vectors on the sampled nodes
# --------------------------------------------------------------------------------------------------------------------------
#
import os
from pathlib import Path

import cv2
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.io as sio
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch_geometric
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def get_vectors(matrix:np.array, idx:np.array , img:np.array, add_pos=False)->torch.Tensor:
    h,w,_ = img.shape
    #Extract and store crops
    crops = []
    for coords in matrix["inst_centroid"][idx,:]:
        y,x = coords
        crop = img[np.clip(int(x-PATCH_SIZE//2),a_min=0,a_max=h):np.clip(int(x+PATCH_SIZE//2),a_min=0,a_max=h),
                np.clip(int(y-PATCH_SIZE//2),a_min=0,a_max=w):np.clip(int(y+PATCH_SIZE//2),a_min=0,a_max=w), :]
        crops.append(transform_deploy(crop))

    #Define dataloader
    dataloader = torch.utils.data.DataLoader(crops, batch_size = BATCH_SIZE)

    with torch.no_grad():
        out = torch.empty((1,512))
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            out = torch.cat((out,output.cpu()))
        out = out[1:,:]
    class_type = torch.eye(CELL_CLASSES)[matrix["inst_type"][idx,0]]
    out = torch.cat((out,class_type),dim=1)
    if add_pos:
        norm_pos = torch.tensor(matrix["inst_centroid"][idx,:]) @ torch.tensor([[1/w,0],[0,1/h]]).double()
        out = torch.cat((out,norm_pos),dim=1)
    return out


def build_graphdata(matrix, img, label, plot=False, overlay_img=None, name=None, add_pos=False):
    """
    Given matrix of metadata from hovernet, constructs graph data
    """
    N = len(matrix['inst_centroid'])
    print(f"Num. of points : {N}")

    #Sampling points using farthest point sampler
    if N > N_SAMPLE:
        point_coords = torch.tensor(matrix["inst_centroid"])
        idx_farthest = farthest_point_sampler(torch.unsqueeze(point_coords, 0), int(SAMPLE_FARTHEST_PERC*N_SAMPLE))[0].numpy()
        all_idx = np.arange(N)
        remaining_idx = np.array(list(set(all_idx)-set(idx_farthest)))
        idx_random = np.random.choice(remaining_idx,int(SAMPLE_RANDOM_PERC*N_SAMPLE),replace=False)
        idx = np.concatenate((idx_farthest,idx_random))
    else:
        idx = np.arange(N)

    #Figure out the best way to make graphs
    adj_matrix = kneighbors_graph(matrix["inst_centroid"][idx,:], 5, mode='distance', include_self=True)
    adj_matrix = adj_matrix.toarray()
    adj_matrix = np.where(adj_matrix<=THRESHOLD,adj_matrix>0,0)

    full_graph = nx.from_numpy_matrix(adj_matrix)

    #Build Feature vectors
    X = get_vectors(matrix, idx , img, add_pos)


    if plot==True:
        print("Plotting graph...")
        img = overlay_img 
        plt.imshow(img)
        pox_x =  matrix['inst_centroid'][idx,0]
        pox_y =  matrix['inst_centroid'][idx,1]
        colors = np.ravel(matrix['inst_type'])[idx]
        plot_coord = {}
        color_map = []
        for i in range(len(pox_x)):
            plot_coord[i] = (pox_x[i], pox_y[i])
            color_map.append(TYPE_INFO[colors[i]])

        plt.ylim(max(pox_y), 0)
        nx.draw(full_graph,plot_coord,node_color=color_map,node_size=5)
        plt.savefig(Path(OUTPUT) / f"{name}.png")

    data = torch_geometric.utils.from_networkx(full_graph)
    data.x = X
    data.y = label
    data.sampled_idx = idx

    return data

#Important constants
THRESHOLD = 500
N_SAMPLE = 3500
SAMPLE_FARTHEST_PERC = 0.75
SAMPLE_RANDOM_PERC = 0.25
#For saving images of graph for visualization, optional
OUTPUT = "../images"

TYPE_INFO = {
    0 : "black", #nolabel
    1 : "red", #neopla
    2 : "green", #inflam
    3 : "blue", #connec
    4 : "yellow", #necros
    5 : "orange" #no-neo
} 

##############################################################################################################################
if __name__ == "__main__":
    from dgl.geometry import farthest_point_sampler
    
    #Initialize network parameters
    # Defining model
    PATCH_SIZE = 72
    RESIZE_DIM = 256
    BATCH_SIZE = 256
    CELL_CLASSES = 6

    GPU_DEVICE = 5
    device = torch.device(f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu")

    transform_deploy = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((128,128)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                        ])
    MODEL_PATH = "../saved_models/simclr/simclr_res34_pretrained.pth"
    state_dict = torch.load(MODEL_PATH,map_location=torch.device("cpu"))
    to_load = {}
    model = torchvision.models.__dict__["resnet34"](pretrained=False)
    model.fc = nn.Sequential()
    for keys,items in state_dict.items():
        if "main_backbone" in keys:
            to_load[keys.replace('main_backbone.', '')]=items
    model.load_state_dict(to_load,strict=False)
    model = model.to(device)
    model.eval()
    ############################################################################################################################
    CLASS_NAME_LIST = ["0_N", "1_PB", "2_UDH", "3_ADH", "4_FEA", "5_DCIS", "6_IC"]
    # mode = "test"
    modes = ["train","val","test"]
    for mode in modes:
        #Path to hovernet results in the form of mat
        path = Path(f"./BRACS_dataset_prev/{mode}")
        #The graph was built on stain normalized images
        img_path = Path(f"./BRACS_prev_stain/{mode}")

        # for class_label in [2,3,4,5,6]:
        for class_label in range(len(CLASS_NAME_LIST)):
            class_name = CLASS_NAME_LIST[class_label]
            print(f"Processing {class_name}")

            folder_name = path / Path(class_name) / "mat"
            graph_path = path / Path(class_name) / "graph_obj_simclr_v2"


            if not graph_path.is_dir():
                os.mkdir(graph_path)

            file_name = list(folder_name.glob("*.mat"))

            #Processed file_list
            processed_file_name = [name_file.stem for name_file in list(graph_path.glob("*.pt"))]
            
            for filename in tqdm(file_name):
                if filename.stem in processed_file_name:
                    continue
                matrix = sio.loadmat(filename)
                img = cv2.cvtColor(cv2.imread(str(img_path / class_name /(filename.stem+".png"))),cv2.COLOR_BGR2RGB)
                data = build_graphdata(matrix, img, class_label, plot=False, overlay_img=None, name=None, add_pos=True)
                torch.save(data, graph_path/(filename.stem+".pt"))