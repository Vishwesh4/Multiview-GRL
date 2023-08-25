from tqdm import tqdm
import torch
import wandb

from ..evaluate import train_logistic_classify, val_logistic_classify, generate_tsnemaps
import trainer

class TrainEngine_multiview(trainer.Trainer):
    def train(self):
        self.model.train()
        for data in tqdm(self.dataset.trainloader):
            data_cell, data_patch = data
            self.optimizer.zero_grad()
            local_emb, global_emb, batch = self.model(data_cell,data_patch,self.device)
            loss = self.model.loss(local_emb, global_emb, batch)
            loss.backward()
            self.optimizer.step()
            # Track loss
            # self.logger.track(loss_value=loss.item())
            # Logging loss
            self.logger.log({"Epoch Train loss": loss.item()})
            if not self.args["SCHEDULER"]["epoch_wise"]:
                self.scheduler.step()

        # self.device2 = torch.device(f"cuda:{self.args['ENGINE']['logreg_device']}")
        self.x_train,self.y_train = self.get_embeddings(dataloader = self.dataset.trainloader)
        acc = train_logistic_classify(self.x_train, self.y_train, self.device)
        # acc = svc_classify(self.x_train, self.y_train)
        self.logger.log({"10-fold accuracy":acc})
 
        if self.current_epoch % 5 == 0:
            fig = generate_tsnemaps(self.x_train, self.y_train, self.args["DATASET"]["class_name_list"])
            self.logger.log({"Train TSNE plot":wandb.Image(fig)})

    def val(self):
        self.x_test,self.y_test = self.get_embeddings(dataloader = self.dataset.testloader)
        preds = val_logistic_classify(self.x_train, self.y_train, self.x_test, self.device)
        # preds = val_svc_classify(self.x_train,self.y_train,self.x_test,self.device)
        self.metrics(preds.to(self.device), self.y_test.to(self.device))
        self.metrics.compute()
        self.metrics.log()

        if self.args["LOGGER"]["use_wandb"]:
            #Plot confusion matrix
            self.logger.log({"conf_mat" : wandb.plot.confusion_matrix(y_true = self.y_test.numpy(),
                                                                    preds = preds.cpu().numpy(),
                                                                    class_names = self.args["DATASET"]["class_name_list"] 
                                                                    )})

        if self.current_epoch % 5 == 0:
            fig = generate_tsnemaps(self.x_test, self.y_test, self.args["DATASET"]["class_name_list"])
            self.logger.log({"Test TSNE plot":wandb.Image(fig)})

        #to not break the code, have to think of some better way than this
        print(self.metrics.results["val_F1Score"])
        return self.metrics.results["val_F1Score"], self.metrics.results["val_F1Score"]

    def get_embeddings(self,dataloader):
        #Perform again to get the embedding for the epoch
        self.model.eval()
        x_data = []
        y_data = []
        with torch.no_grad():
            for data in dataloader:
                # store embedding of training data
                data_cell, data_patch = data
                graph_embeddings = self.model.get_representation(data_cell, data_patch, self.device)
                x_data.append(graph_embeddings.cpu())
                y_data.append(data_cell.y)

        x_data = torch.concat(x_data)
        y_data = torch.ravel(torch.concat(y_data))
        return x_data, y_data