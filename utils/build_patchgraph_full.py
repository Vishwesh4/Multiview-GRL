# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is about making patch graphs using output mat from hovernet model, Here I have used KNN neighbour. 
# Basically selects its 8 surrounding neighbours
# --------------------------------------------------------------------------------------------------------------------------
import os
from pathlib import Path

import cv2
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch_geometric
from sklearn.neighbors import kneighbors_graph
from matplotlib import pyplot as plt

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def get_vectors(img:np.array, add_pos=False)->torch.Tensor:
    h,w,_ = img.shape
    #Extract and store crops
    crops = []
    pos_index = []
    patch_size = PATCH_SIZE
    patch_size_x=patch_size
    patch_size_y=patch_size
    if h/patch_size_x < 2:
        patch_size_x = h
        buffer_x=0
    else:
        #To get equally spaced patches
        buffer_x = float((h-patch_size_x*np.floor(h/patch_size_x))/(np.floor(h/patch_size_x)-1))
    if w/patch_size_y < 2:
        patch_size_y = w
        buffer_y=0
    else:
        #To get equally spaced patches
        buffer_y = float((w-patch_size_y*np.floor(w/patch_size_y))/(np.floor(w/patch_size_y)-1))

    for i,x in enumerate(range(0,h-patch_size_x+1,patch_size_x)):
        for j,y in enumerate(range(0,w-patch_size_y+1,patch_size_y)):
            x_idx = int(x+i*buffer_x)
            y_idx = int(y+j*buffer_y)
            img_crop = img[x_idx:x_idx+patch_size_x,y_idx:y_idx+patch_size_y,:]
            crops.append(transform_deploy(img_crop))
            pos_index.append((y_idx+float(patch_size_y)/2,x_idx+float(patch_size_x)/2))
    #Define dataloader
    dataloader = torch.utils.data.DataLoader(crops, batch_size = BATCH_SIZE)

    with torch.no_grad():
        out = torch.empty((1,512))
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            out = torch.cat((out,output.cpu()))
        out = out[1:,:]
    if add_pos:
        norm_pos = torch.tensor(np.array(pos_index)) @ torch.tensor([[1/w,0],[0,1/h]]).double()
        out = torch.cat((out,norm_pos),dim=1)
    return out, np.array(pos_index)

def build_patchgraph(img, label, plot=False, name=None, add_pos=False):
    """
    Builds patch graph
    """
    #Build Feature vectors
    X,centroids = get_vectors(img, add_pos)
    N = len(centroids)
    print(f"Num. of points : {N}")
    #For now skip graphs with a single node
    if N==1:
        return None

    n_neighbours = np.min((8,N-1))

    try:
        dist_x = centroids[1,0] - centroids[0,0]
        dist_y = centroids[np.where(centroids[:,0]==centroids[0,0])[0][1],1] - centroids[0,1]
        dist = np.sqrt(dist_x**2+dist_y**2) + 10
    except:
        dist = 2*PATCH_SIZE
    #Figure out the best way to make graphs
    adj_matrix = kneighbors_graph(centroids, n_neighbours, mode='distance', include_self=False)
    adj_matrix = adj_matrix.toarray()
    adj_matrix = np.where(adj_matrix<dist,adj_matrix>0,0)
    full_graph = nx.from_numpy_matrix(adj_matrix)

    if plot==True:
        print("Plotting graph...")
        plt.imshow(img)
        pox_x =  centroids[:,0]
        pox_y =  centroids[:,1]
        plot_coord = {}
        for i in range(len(pox_x)):
            plot_coord[i] = (pox_x[i], pox_y[i])

        # plt.ylim(max(pox_y), 0)
        nx.draw(full_graph,plot_coord,node_size=5)
        plt.savefig(Path(OUTPUT) / f"{name}.png")

    data = torch_geometric.utils.from_networkx(full_graph)
    data.x = X
    data.y = label

    return data


##############################################################################################################################
if __name__ == "__main__":
    
    #Saving images for visualizing graph, optional
    OUTPUT = "../images"
    #Initialize network parameters
    # Defining model
    PATCH_SIZE = 128 #smaller patch size
    BATCH_SIZE = 256

    GPU_DEVICE = 5
    device = torch.device(f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu")

    transform_deploy = transforms.Compose([
                                            transforms.ToPILImage(),
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
    modes = ["train","val","test"]
    for mode in modes:
        #Path to hovernet results in the form of mat
        path = Path(f"./BRACS_dataset_prev/{mode}")
        #The graph was built on stain normalized images
        img_path = Path(f"./BRACS_prev_stain/{mode}")

        for class_label in range(len(CLASS_NAME_LIST)):
            class_name = CLASS_NAME_LIST[class_label]
            print(f"Processing {class_name}")

            folder_name = img_path / class_name 
            graph_path = path / class_name / "patch_graph_custom_simclr"

            if not graph_path.is_dir():
                os.mkdir(graph_path)

            file_name = list(folder_name.glob("*.png"))

            #Processed file_list
            processed_file_name = [name_file.stem for name_file in list(graph_path.glob("*.pt"))]

            for filename in tqdm(file_name):
                if filename.stem in processed_file_name:
                    continue
                img = cv2.cvtColor(cv2.imread(str(img_path / class_name /(filename.stem+".png"))),cv2.COLOR_BGR2RGB)
                data = build_patchgraph(img, class_label, plot=False, name=None, add_pos=True)
                if data is not None:
                    torch.save(data, graph_path/(filename.stem+".pt"))