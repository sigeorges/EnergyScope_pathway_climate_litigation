# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:25:38 2023

@author: julia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%% MIX ENERGETIQUE

df = pd.read_excel("output_77k_55_500/output_77k_55%_500/resources.xlsx", sheet_name = 1).set_index("Name")

colors = ["gold","skyblue", "peru", "burlywood", "teal", "olive", "darkkhaki", "orange", "palegreen", "navy", "cornflowerblue", "pink", "lightcoral", "violet", "darkmagenta","deeppink"]

sns.set_style("ticks")

df.T.plot.bar(stacked=True, color = colors)
sns.despine()
plt.xticks(rotation=0)
sns.set(font_scale=2)


solid = []
oil = []
NG = []
waste = [] 
res = []


#%%
folder = "output_77k_60_1k/output_77k_60_1k/"
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
co2net_limit = [77000, 61849.375, 46698.75, 35149.0625, 23599.375, 12049.6875, 500]
co2net = []
with open(folder + "CO2NET_YEARLY.txt") as file : 
    lines = file.readline()
    lines = file.readline()
    co2 = lines.split(" ")
    for i in range(1, len(co2)-1) : 
        co2net.append(float(co2[i]))

plt.plot(years, co2net_limit, color = 'firebrick', label='limit')
plt.scatter(years, co2net_limit, color = 'firebrick')
plt.plot(years, co2net, color = 'navy', label='net')
plt.scatter(years, co2net, color = 'navy')
plt.legend()
plt.show()

#%%
folder = "output_77k_55_500/output_77k_55%_500/elec_share.xlsx"
df = pd.read_excel(folder).set_index("Tech")
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

lst = []
for i in range(len(df)) : 
    obj = df.iloc[i]
    todel = True
    for j in range(len(obj.values)) : 
        if obj.values[j] > 1 : 
            todel = False
    if todel : 
        lst.append(i)
        
df = df.drop(df.index[lst])

colors = ["gold","skyblue", "peru", "burlywood", "teal", "olive", "darkkhaki", "orange", "palegreen", "navy", "cornflowerblue", "pink", "lightcoral", "violet", "darkmagenta","deeppink"]
interesting = [' NUCLEAR ', " CCGT ", " IND_COGEN_GAS ", " WIND_ONSHORE ", " WIND_OFFSHORE ", " PV ", " CCGT_AMMONIA "]

others = [0, 0, 0, 0, 0, 0, 0]
for y in range(len(years)) : 
    for i in range(len(df)) : 
        name = df.index[i]
        if name not in interesting : 
            print(name)
            others[y] += df.iloc[i,y]
print(others)

for i in range(len(interesting)) : 
    if df.index[i] in interesting : 
        plt.plot(years, df.iloc[i], label=df.index[i], color=colors[i])
plt.plot(years, others, label='others', color = colors[-1])
plt.legend()
plt.show()

#%%
folder = "output_77k_55_500/output_77k_55%_500/mob_share.xlsx"
df = pd.read_excel(folder).set_index("Tech")
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

lst = []
for i in range(len(df)) : 
    obj = df.iloc[i]
    todel = True
    for j in range(len(obj.values)) : 
        if obj.values[j] > 1 : 
            todel = False
    if todel : 
        lst.append(i)

df = df.drop(df.index[lst])

lst_freight = []
lst_pass = []
for i in range(len(df)) : 
    name = df.index[i]
    if "FREIGHT" in name or "TRUCK" in name :
        lst_freight.append(i)
    else : 
        lst_pass.append(i)

df_freight = df.drop(df.index[lst_pass])
df_pass = df.drop(df.index[lst_freight])

colors = ["skyblue", "teal", "orange", "palegreen", "cornflowerblue", "pink", "violet", "darkmagenta","deeppink"]

for i in range(len(df_freight)) : 
    plt.plot(years, df_freight.iloc[i], label=df_freight.index[i], color = colors[i])
plt.legend()
plt.show()
#%%
for i in range(len(df_pass)) : 
    plt.plot(years, df_pass.iloc[i], label=df_pass.index[i], color = colors[i])
plt.legend()
plt.show()

#%%
folder = "output_77k_55_500/output_77k_55%_500/HT_share.xlsx"
df = pd.read_excel(folder).set_index("Tech")
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

lst = []
for i in range(len(df)) : 
    obj = df.iloc[i]
    todel = True
    for j in range(len(obj.values)) : 
        if obj.values[j] > 1 : 
            todel = False
    if todel : 
        lst.append(i)

df = df.drop(df.index[lst])

colors = ["skyblue", "teal", "orange", "palegreen", "cornflowerblue", "pink", "darkmagenta","deeppink"]
for i in range(len(df)) : 
    plt.plot(years, df.iloc[i], label=df.index[i], color = colors[i])
plt.legend()
plt.show()
#%%
folder = "output_77k_55_500/output_77k_55%_500/LTdec_share.xlsx"
df = pd.read_excel(folder).set_index("Tech")
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

lst = []
for i in range(len(df)) : 
    obj = df.iloc[i]
    todel = True
    for j in range(len(obj.values)) : 
        if obj.values[j] > 1 : 
            todel = False
    if todel : 
        lst.append(i)

df = df.drop(df.index[lst])

colors = ["skyblue", "teal", "orange", "palegreen", "cornflowerblue", "pink", "darkmagenta","deeppink"]
for i in range(len(df)) : 
    plt.plot(years, df.iloc[i], label=df.index[i], color = colors[i])
plt.legend()
plt.show()

#%%
folder = "output_77k_55_500/output_77k_55%_500/LTdhn_share.xlsx"
df = pd.read_excel(folder).set_index("Tech")
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

lst = []
for i in range(len(df)) : 
    obj = df.iloc[i]
    todel = True
    for j in range(len(obj.values)) : 
        if obj.values[j] > 1 : 
            todel = False
    if todel : 
        lst.append(i)

df = df.drop(df.index[lst])

colors = ["skyblue", "teal", "orange", "palegreen", "cornflowerblue", "pink", "darkmagenta","deeppink"]
for i in range(len(df)) : 
    plt.plot(years, df.iloc[i], label=df.index[i], color = colors[i])
plt.legend()
plt.show()
