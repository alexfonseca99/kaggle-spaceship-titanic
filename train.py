import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# Ideias:
# Binarizar clases categóricas
# Não utilizar elementos incompletos


class Predictor(torch.nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()

        # Binarized classes = 27
        # Non-binarized = 13
        self.linear1 = torch.nn.Linear(27, 100)
        self.linear2 = torch.nn.Linear(100, 200)
        self.linear3 = torch.nn.Linear(200, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.softmax(x)
        return x


def main():

    d = pd.read_csv("train.csv")

    #  Pre-process data types. "Cabin" is split into "Deck", "Number" and "Side"
    # d["HomePlanet"] = pd.Categorical(d["HomePlanet"]).codes
    # d["CryoSleep"] = pd.Categorical(d["CryoSleep"]).codes
    # d["Destination"] = pd.Categorical(d["Destination"]).codes
    # d["VIP"] = pd.Categorical(d["VIP"]).codes
    # d["Deck"] = pd.Categorical(d["Cabin"].str.split("/").str.get(0)).codes
    # d["Number"] = pd.to_numeric(d["Cabin"].str.split("/").str.get(1))
    # d["Side"] = pd.Categorical(d["Cabin"].str.split("/").str.get(2)).codes
    #
    # #  Fill missing data
    # d["Age"].fillna(d["Age"].median(), inplace=True)
    # d["Number"].fillna(d["Number"].median(), inplace=True)
    # d["RoomService"].fillna(d["RoomService"].median(), inplace=True)
    # d["FoodCourt"].fillna(d["FoodCourt"].median(), inplace=True)
    # d["ShoppingMall"].fillna(d["ShoppingMall"].median(), inplace=True)
    # d["Spa"].fillna(d["Spa"].median(), inplace=True)
    # d["VRDeck"].fillna(d["VRDeck"].median(), inplace=True)
    # d = d[["HomePlanet", "CryoSleep", "Destination", "Age", "VIP", "RoomService",
    #        "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Deck", "Number", "Side", "Transported"]]

    d[["Earth", "Europa", "Mars", "UnknownPlanet"]] = OneHotEncoder().fit_transform(d["HomePlanet"].fillna("U").
                                                                                    to_numpy().reshape(-1, 1)).toarray()
    d[["Cancri", "PSO", "TRAPPIST", "UnknownDest"]] = OneHotEncoder().fit_transform(d["Destination"].fillna("U").
                                                                                    to_numpy().reshape(-1, 1)).toarray()
    d[["A", "B", "C", "D", "E", "F", "G", "T", "U"]] = OneHotEncoder().fit_transform(
        d["Cabin"].str.split("/").str.get(0).fillna("U").to_numpy().reshape(-1, 1)).toarray()

    d["CryoSleep"] = pd.Categorical(d["CryoSleep"]).codes
    d["VIP"] = pd.Categorical(d["VIP"]).codes
    d["Number"] = pd.to_numeric(d["Cabin"].str.split("/").str.get(1))
    d["Side"] = pd.Categorical(d["Cabin"].str.split("/").str.get(2)).codes
    d["Transported"] = d["Transported"].replace({True: 1, False: 0})

    #  Drop missing data
    d = d.dropna()
    #  Fill missing data
    # d["Age"].fillna(d["Age"].median(), inplace=True)
    # d["Number"].fillna(d["Number"].median(), inplace=True)
    # d["RoomService"].fillna(d["RoomService"].median(), inplace=True)
    # d["FoodCourt"].fillna(d["FoodCourt"].median(), inplace=True)
    # d["ShoppingMall"].fillna(d["ShoppingMall"].median(), inplace=True)
    # d["Spa"].fillna(d["Spa"].median(), inplace=True)
    # d["VRDeck"].fillna(d["VRDeck"].median(), inplace=True)

    d = d[["CryoSleep", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Number", "Side",
           "Earth", "Europa", "Mars", "UnknownPlanet", "Cancri", "PSO", "TRAPPIST", "UnknownDest", "A", "B", "C", "D",
           "E", "F", "G", "T", "U", "Transported"]]

    #  Normalize data
    scaler = StandardScaler()
    cols_to_scale = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Number"]
    d[cols_to_scale] = scaler.fit_transform(d[cols_to_scale])

    #  Split train/validation data and labels
    #  0 - Not Transported, 1 - Transported
    train_data, val_data = train_test_split(d, test_size=0.2)
    train_labels = train_data["Transported"].to_numpy()
    val_labels = val_data["Transported"].to_numpy()
    train_data = train_data.drop(columns="Transported").values
    val_data = val_data.drop(columns="Transported").values
    train_labels = OneHotEncoder().fit(train_labels.reshape(-1, 1)).transform(train_labels.reshape(-1, 1)).toarray()
    val_labels = OneHotEncoder().fit(val_labels.reshape(-1, 1)).transform(val_labels.reshape(-1, 1)).toarray()

    #  Convert to tensor and create DataLoader
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    val_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    #  Create model, loss function and optimizer
    model = Predictor()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #  Training loop. Set number of epochs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_epoch = 0
    epochs = 20
    best_val_loss = 1e6

    for epoch in range(epochs):
        model.train(True)
        print(f"******\t EPOCH {epoch} \t******")
        #  Train one epoch
        running_loss = 0.
        running_vloss = 0.
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            out = model(inputs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            #  Check loss values during training
            running_loss += loss.item()
            """
            if i % 20 == 0:
                last_loss = running_loss / 20
                print(f"batch {i+1}\tloss{last_loss}")
                running_loss = 0.
            """
        #  Average training loss
        avg_loss = running_loss / (i + 1)

        #  Validation mode
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                vout = model(vinputs)
                vloss = loss_fn(vout, vlabels)
                running_vloss += vloss
        #  Average validation loss
        avg_val_loss = running_vloss / (i + 1)


        print(f"LOSS train {avg_val_loss}\trunning loss {avg_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"model_{timestamp}"
            torch.save(model.state_dict(), model_path)

    return


if __name__ == '__main__':
    main()