#!/usr/bin/env python3

import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

df = pd.read_csv("games-features.csv")

fields = [
    "Metacritic",
    "ReleaseDate"
]

# for field in fields:
#     df1 = df[field][getattr(df, field) > 2]
#     ax = df1.plot.hist(by=field, bins=75)
#     ax.plot()
#     plt.savefig("hist_{}.png".format(field))

df2 = df["Metacritic"][df.Metacritic > 1]
ax = df2.plot.hist(by="Metacritic", bins=75)
ax.set_xlabel("Metacritic score")
ax.set_title("Metacritic score distribution")
ax.plot()
plt.savefig("hist_Metacritic.png")

# print(df.loc[0])
