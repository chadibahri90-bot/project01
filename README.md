#partieA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv ( "electricite_ete_2026.csv" )

print("Les 10 premières lignes :")
display(df.head(10))

print("Les 5 dernières lignes :")
display(df.tail(5))

print("Colonnes :")
print(df.columns)

print("\nTypes des données :")
print(df.dtypes)

print("\nDimensions :")
print(df.shape)

print("\nNombre total d'éléments :")
print(df.size)

#partieB


print(df.describe())

moyenne = df["Consommation"].mean()
mediane = df["Consommation"].median()
ecart_type = df["Consommation"].std()
minimum = df["Consommation"].min()
maximum = df["Consommation"].max()

print("Moyenne :", moyenne)
print("Médiane :", mediane)
print("Écart-type :", ecart_type)
print("Minimum :", minimum)
print("Maximum :", maximum)

jours_chauds = df[df["Temperature"] > 35]

print(jours_chauds)

moyenne_consommation = df["Consommation"].mean()

jours_consommation_elevee = df[df["Consommation"] > moyenne_consommation]

print("Moyenne de consommation :", moyenne_consommation)
print(jours_consommation_elevee)

df_trie = df.sort_values(by="Consommation",ascending=False)

print(df_trie)

moyenne_coupure = df.groupby("Coupure")["Consommation"].mean()

print(moyenne_coupure)

#partieC

plt.figure(figsize=(9, 4))

plt.hist(df["Temperature"], bins = 20)
plt.title("Distribution de la température")
plt.xlabel("Température (°C)")
plt.ylabel("Fréquence")

plt.show()

df["Date"] = pd.to_datetime(df["Date"])
plt.figure(figsize=(9,4))

plt.plot(
    df["Date"],
    df["Consommation"],
    label="Consommation"
)

plt.title("Évolution de la consommation électrique")
plt.xlabel("Date")
plt.ylabel("Consommation (MWh/jour)")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter( df["Temperature"], df["Consommation"])

plt.title("Relation entre température et consommation")
plt.xlabel("Température (°C)")
plt.ylabel("Consommation (MWh/jour)")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(x="Coupure",y="Consommation",data=df)

plt.title("Consommation selon la présence d'une coupure")
plt.xlabel("Coupure")
plt.ylabel("Consommation (MWh/jour)")

plt.show()

#partieD

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["Coupure_encoded"] = encoder.fit_transform(df["Coupure"])
y = df["Coupure_encoded"]
print(y)

X = df[["Temperature", "Humidite", "Consommation"]]
y = df["Coupure_encoded"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

print("Données d'entraînement :")
print(X_train)
print("Données de test :")
print(X_test)

print("Taille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test :", y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)
print("Précision en pourcentage :", accuracy * 100, "%")

#partieE

