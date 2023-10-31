import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # HomePlanet, Destination, Deck (cabin letter)
    d = pd.read_csv("train.csv")
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
           "E", "F", "G", "T", "U", "Transported"]]

    #  Normalize data
    scaler = StandardScaler()
    cols_to_scale = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Number"]
    d[cols_to_scale] = scaler.fit_transform(d[cols_to_scale])

    corr = d.corr()
    f, ax = plt.subplots(figsize=(20,14))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap)
    plt.title("fill na")
    plt.show()

    return


if __name__ == "__main__":
    main()