import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from train import Predictor


def main():

    d = pd.read_csv("test.csv")
    ids = pd.DataFrame(d["PassengerId"])
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
    #        "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Deck", "Number", "Side"]]

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

    #  Drop missing data
    # d = d.dropna()
    #  Fill missing data
    d["Age"].fillna(d["Age"].median(), inplace=True)
    d["Number"].fillna(d["Number"].median(), inplace=True)
    d["RoomService"].fillna(d["RoomService"].median(), inplace=True)
    d["FoodCourt"].fillna(d["FoodCourt"].median(), inplace=True)
    d["ShoppingMall"].fillna(d["ShoppingMall"].median(), inplace=True)
    d["Spa"].fillna(d["Spa"].median(), inplace=True)
    d["VRDeck"].fillna(d["VRDeck"].median(), inplace=True)

    d = d[["CryoSleep", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Number", "Side",
           "Earth", "Europa", "Mars", "UnknownPlanet", "Cancri", "PSO", "TRAPPIST", "UnknownDest", "A", "B", "C", "D",
           "E", "F", "G", "T", "U"]]

    #  Normalize data
    scaler = StandardScaler()
    cols_to_scale = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Number"]
    d[cols_to_scale] = scaler.fit_transform(d[cols_to_scale])
    test_data = torch.Tensor(d.values)

    model = Predictor()
    model.load_state_dict(torch.load("model_binarized_labels_dropna"))

    out = torch.argmax(model(test_data), axis=1)

    ids["Transported"] = pd.Series(out.numpy()).astype('bool')
    ids.to_csv("submission_binarized_labels_dropna.csv", index=False)

    return


if __name__ == "__main__":
    main()