# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:16:31 2023

@author: Simon Georges (UCLouvain)
"""
# %% Packages
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import os
# %% Encoding repositery path

folder = '..\STEP_2_Pathway_Model'
path = folder + '\output'
graph_path = folder + '\graphs'

years = np.linspace(2020, 2050, 7, dtype=int).astype(str)

# %% Useful secondary functions
def series2Dic(series, cond):
    dic = {}
    for el in series.index:
        if cond == 'pos':
            if series[el] > 0:
                dic[el] = series[el]
        elif cond == 'neg':
            if series[el] < 0:
                dic[el] = -series[el]
    return dic

# %% Main functions
def total_costs(path, years, plot=False):
    
    chf2eur = 1.02 #[CHF/EUR]
    
    yearly_costs = {}
    
    for year in years:
        dir_name = '\YEAR_' + year
        
        costs_df = pd.read_table(path+dir_name+'\cost_breakdown.txt', index_col='Name')
        
        cost_dic = {}
        for col in costs_df.columns:
            cost_dic[col] = sum(costs_df[col])*chf2eur
        
        yearly_costs[year] = cost_dic
        
    trans_costs = {}
    
    for sector in yearly_costs[years[0]].keys():
        sector_costs = [yearly_costs[year][sector] for year in years]
        trans_costs[sector] = sector_costs
    
    plot_df = pd.DataFrame(trans_costs)
    plot_df['YEARS'] = years
    
    y_plot = ['C_inv', 'C_maint', 'C_op']
    
    if plot == True:
        fig = px.area(plot_df, x='YEARS', y=y_plot)
    else:
        fig = None
    
    return (plot_df, trans_costs, fig)


def costs_per_sector(path, years, plot=False, cost_type=['C_inv', 'C_maint', 'C_op'], sectors = ['ELECTRICITY', 'HEAT_HT', 'HEAT_DHN', 'HEAT_DEC', 'MOB_PUBLIC', 'MOB_PRIVATE', 'MOB_FREIGHT', 'SYNTHETIC_FUELS', 'STORAGE', 'INFRASTRUCTURE', 'RENEW_RES', 'NON_RENEW_RES'], norm=False):
    
    chf2eur = 1.02 #[CHF/EUR]
    chp_elec_part = 0.5 #[Wh_elec/Wh_tot]
    
    yearly_costs = {}
    
    prefix_dic = {'ELECTRICITY': ['NUCLEAR', 'CCGT', 'CCGT_AMMONIA', 'COAL_US', 'COAL_IGCC', 'PV', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO_RIVER', 'GEOTHERMAL'],
               'HEAT_HT': ['IND_COGEN_GAS', 'IND_COGEN_WOOD', 'IND_COGEN_WASTE', 'IND_BOILER_GAS', 'IND_BOILER_WOOD', 'IND_BOILER_OIL', 'IND_BOILER_COAL', 'IND_BOILER_WASTE', 'IND_DIRECT_ELEC'],
               'HEAT_DHN': ['DHN_HP_ELEC', 'DHN_COGEN_GAS', 'DHN_COGEN_WOOD', 'DHN_COGEN_WET_BIOMASS', 'DHN_COGEN_BIO_HYDROLYSIS', 'DHN_COGEN_WASTE', 'DHN_BOILER_GAS', 'DHN_BOILER_WOOD', 'DHN_BOILER_OIL', 'DHN_DEEP_GEO', 'DHN_SOLAR'],
               'HEAT_DEC': ['DEC_HP_ELEC', 'DEC_THHP_GAS', 'DEC_COGEN_GAS', 'DEC_COGEN_OIL', 'DEC_ADVCOGEN_GAS', 'DEC_ADVCOGEN_H2', 'DEC_BOILER_GAS', 'DEC_BOILER_WOOD', 'DEC_BOILER_OIL', 'DEC_SOLAR', 'DEC_DIRECT_ELEC'],
               'MOB_PUBLIC': ['TRAMWAY_TROLLEY', 'BUS_COACH_DIESEL', 'BUS_COACH_HYDIESEL', 'BUS_COACH_CNG_STOICH', 'BUS_COACH_FC_HYBRIDH2', 'TRAIN_PUB'],
               'MOB_PRIVATE': ['CAR_GASOLINE', 'CAR_DIESEL', 'CAR_NG', 'CAR_METHANOL', 'CAR_HEV', 'CAR_PHEV', 'CAR_BEV', 'CAR_FUEL_CELL'],
               'MOB_FREIGHT': ['TRAIN_FREIGHT', 'BOAT_FREIGHT_DIESEL', 'BOAT_FREIGHT_NG', 'BOAT_FREIGHT_METHANOL', 'TRUCK_DIESEL', 'TRUCK_FUEL_CELL', 'TRUCK_NG', 'TRUCK_METHANOL', 'TRUCK_ELEC'],
               'SYNTHETIC_FUELS': ['HABER_BOSCH', 'OIL_TO_HVC', 'GAS_TO_HVC', 'BIOMASS_TO_HVC', 'METHANOL_TO_HVC', 'SYN_METHANOLATION', 'METHANE_TO_METHANOL', 'BIOMASS_TO_METHANOL', 'H2_ELECTROLYSIS', 'SMR', 'H2_BIOMASS', 'GASIFICATION_SNG', 'SYN_METHANATION', 'BIOMETHANATION', 'BIO_HYDROLYSIS', 'PYROLYSIS_TO_LFO', 'PYROLYSIS_TO_FUELS', 'ATM_CCS', 'INDUSTRY_CCS', 'AMMONIA_TO_H2'],
               'STORAGE': ['PHS', 'BATT_LI', 'BEV_BATT', 'PHEV_BATT', 'TS_DEC_HP_ELEC', 'TS_DEC_DIRECT_ELEC', 'TS_DHN_DAILY', 'TS_DHN_SEASONAL', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL', 'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL', 'TS_HIGH_TEMP', 'GAS_STORAGE', 'H2_STORAGE', 'CO2_STORAGE', 'GASOLINE_STORAGE', 'DIESEL_STORAGE', 'METHANOL_STORAGE', 'AMMONIA_STORAGE', 'LFO_STORAGE'],
               'INFRASTRUCTURE': ['EFFICIENCY', 'DHN', 'GRID'],
               'RENEW_RES': ['BIOETHANOL', 'BIODIESEL', 'GAS_RE', 'WOOD', 'WET_BIOMASS', 'WASTE', 'H2', 'ELECTRICITY', 'METHANOL_RE', 'AMMONIA_RE'],
               'NON_RENEW_RES': ['GASOLINE', 'DIESEL', 'LFO', 'GAS', 'COAL', 'URANIUM', 'AMMONIA', 'METHANOL'],
               'ALL_RES': ['ELECTRICITY', 'GASOLINE', 'DIESEL', 'LFO', 'GAS', 'GAS_RE', 'COAL', 'URANIUM', 'WOOD', 'WET_BIOMASS', 'WASTE', 'H2', 'AMMONIA', 'AMMONIA_RE', 'METHANOL', 'METHANOL_RE', 'BIOETHANOL', 'BIODIESEL']}
    
    used_sect = {key: prefix_dic[key] for key in prefix_dic.keys() if key in sectors}
    
    for year in years:
        dir_name = '\YEAR_' + year
        
        costs_df = pd.read_table(path+dir_name+'\cost_breakdown.txt', index_col='Name')
        
        cost_dic = {key: 0 for key in used_sect.keys()}
        for sector in used_sect.keys():
            sect = sector
            if 'd' in sector.split('_'):
                sect = ('_').join(sector.split('_')[:-1])
            for el in used_sect[sector]:
                to_add = sum([costs_df.loc[el][type] for type in cost_type])
                if 'COGEN' in el.split('_'):
                    chp_cost = to_add*chf2eur
                    cost_dic['ELECTRICITY'] += chp_cost*chp_elec_part
                    if sect != 'ELECTRICITY':
                        cost_dic[sect] += chp_cost*(1-chp_elec_part)
                else:
                     cost_dic[sect] += to_add*chf2eur
        yearly_costs[year] = cost_dic
    trans_costs = {}
    
    for sector in yearly_costs[years[0]].keys():
        sector_costs = [yearly_costs[year][sector] for year in years]
        trans_costs[sector] = sector_costs
    
    plot_df = pd.DataFrame(trans_costs)
    
    if norm:
        yearly_tot = plot_df[sectors].sum(axis=1) #Shares of partial system demand (norm wrt selected sectors sum)
        yearly_tot = plot_df.sum(axis=1) #Shares of global system demand (norm wrt all sectors sum)
        for sector in sectors:
            plot_df[sector] = plot_df[sector]/yearly_tot
    
    EUD_colors = {'HEAT_DEC': "#FF9999", 'HEAT_DHN': "firebrick", 'HEAT_HT': "#FF0000", 'ELECTRICITY': "#00B0F0",
                  'MOB_FREIGHT': "#996633", 'MOB_PUBLIC': "#FFC000", 'MOB_PRIVATE': "#BF9000"}
    
    to_melt = plot_df[sectors].copy(deep=True)
    to_melt['YEARS'] = years
    
    melt_plot = pd.melt(to_melt, id_vars='YEARS', var_name='Sector', value_name='Cost')
    
    if plot == True:
        fig = px.area(melt_plot, x='YEARS', y='Cost', color='Sector', color_discrete_map=EUD_colors)
    else:
        fig = None
    
    return (plot_df, trans_costs, fig)


def sector_costs(path, years, sectors=['ELECTRICITY', 'HEAT_HT', 'HEAT_DHN', 'HEAT_DEC', 'MOB_PUBLIC', 'MOB_PRIVATE', 'MOB_FREIGHT', 'SYNTHETIC_FUELS', 'STORAGE', 'INFRASTRUCTURE', 'RENEW_RES', 'NON_RENEW_RES'], plot=False, norm=False):
    
    chf2eur = 1.02 #[CHF/EUR]
    chp_elec_part = 0.5 #[Wh_elec/Wh_tot]
    
    prefix_dic = {'ELECTRICITY': ['NUCLEAR', 'CCGT', 'CCGT_AMMONIA', 'COAL_US', 'COAL_IGCC', 'PV', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO_RIVER', 'GEOTHERMAL'],
               'HEAT_HT': ['IND_COGEN_GAS', 'IND_COGEN_WOOD', 'IND_COGEN_WASTE', 'IND_BOILER_GAS', 'IND_BOILER_WOOD', 'IND_BOILER_OIL', 'IND_BOILER_COAL', 'IND_BOILER_WASTE', 'IND_DIRECT_ELEC'],
               'HEAT_DHN': ['DHN_HP_ELEC', 'DHN_COGEN_GAS', 'DHN_COGEN_WOOD', 'DHN_COGEN_WET_BIOMASS', 'DHN_COGEN_BIO_HYDROLYSIS', 'DHN_COGEN_WASTE', 'DHN_BOILER_GAS', 'DHN_BOILER_WOOD', 'DHN_BOILER_OIL', 'DHN_DEEP_GEO', 'DHN_SOLAR'],
               'HEAT_DEC': ['DEC_HP_ELEC', 'DEC_THHP_GAS', 'DEC_COGEN_GAS', 'DEC_COGEN_OIL', 'DEC_ADVCOGEN_GAS', 'DEC_ADVCOGEN_H2', 'DEC_BOILER_GAS', 'DEC_BOILER_WOOD', 'DEC_BOILER_OIL', 'DEC_SOLAR', 'DEC_DIRECT_ELEC'],
               'MOB_PUBLIC': ['TRAMWAY_TROLLEY', 'BUS_COACH_DIESEL', 'BUS_COACH_HYDIESEL', 'BUS_COACH_CNG_STOICH', 'BUS_COACH_FC_HYBRIDH2', 'TRAIN_PUB'],
               'MOB_PRIVATE': ['CAR_GASOLINE', 'CAR_DIESEL', 'CAR_NG', 'CAR_METHANOL', 'CAR_HEV', 'CAR_PHEV', 'CAR_BEV', 'CAR_FUEL_CELL'],
               'MOB_FREIGHT': ['TRAIN_FREIGHT', 'BOAT_FREIGHT_DIESEL', 'BOAT_FREIGHT_NG', 'BOAT_FREIGHT_METHANOL', 'TRUCK_DIESEL', 'TRUCK_FUEL_CELL', 'TRUCK_NG', 'TRUCK_METHANOL', 'TRUCK_ELEC'],
               'SYNTHETIC_FUELS': ['HABER_BOSCH', 'OIL_TO_HVC', 'GAS_TO_HVC', 'BIOMASS_TO_HVC', 'METHANOL_TO_HVC', 'SYN_METHANOLATION', 'METHANE_TO_METHANOL', 'BIOMASS_TO_METHANOL', 'H2_ELECTROLYSIS', 'SMR', 'H2_BIOMASS', 'GASIFICATION_SNG', 'SYN_METHANATION', 'BIOMETHANATION', 'BIO_HYDROLYSIS', 'PYROLYSIS_TO_LFO', 'PYROLYSIS_TO_FUELS', 'ATM_CCS', 'INDUSTRY_CCS', 'AMMONIA_TO_H2'],
               'STORAGE': ['PHS', 'BATT_LI', 'BEV_BATT', 'PHEV_BATT', 'TS_DEC_HP_ELEC', 'TS_DEC_DIRECT_ELEC', 'TS_DHN_DAILY', 'TS_DHN_SEASONAL', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL', 'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL', 'TS_HIGH_TEMP', 'GAS_STORAGE', 'H2_STORAGE', 'CO2_STORAGE', 'GASOLINE_STORAGE', 'DIESEL_STORAGE', 'METHANOL_STORAGE', 'AMMONIA_STORAGE', 'LFO_STORAGE'],
               'INFRASTRUCTURE': ['EFFICIENCY', 'DHN', 'GRID'],
               'RENEW_RES': ['BIOETHANOL', 'BIODIESEL', 'GAS_RE', 'WOOD', 'WET_BIOMASS', 'WASTE', 'H2', 'ELECTRICITY', 'METHANOL_RE', 'AMMONIA_RE'],
               'NON_RENEW_RES': ['GASOLINE', 'DIESEL', 'LFO', 'GAS', 'COAL', 'URANIUM', 'AMMONIA', 'METHANOL'],
               'ALL_RES': ['ELECTRICITY', 'GASOLINE', 'DIESEL', 'LFO', 'GAS', 'GAS_RE', 'COAL', 'URANIUM', 'WOOD', 'WET_BIOMASS', 'WASTE', 'H2', 'AMMONIA', 'AMMONIA_RE', 'METHANOL', 'METHANOL_RE', 'BIOETHANOL', 'BIODIESEL']}
    
    used_sect = {key: prefix_dic[key] for key in prefix_dic.keys() if key in sectors}
    
    yearly_costs = {}
    
    for year in years:
        
        dir_name = '\YEAR_' + year
        
        costs_df = pd.read_table(path+dir_name+'\cost_breakdown.txt', index_col='Name')
        cost_dic = {}
        for sector in sectors:
            sector_dic = {key: 0 for key in prefix_dic[sector]}
            for el in prefix_dic[sector]:
                if 'COGEN' in el.split('_'):
                    if sector == 'ELECTRICITY':
                        to_add = costs_df.loc[el].sum()*chp_elec_part
                    else:
                        to_add = costs_df.loc[el].sum()*(1-chp_elec_part)
                else:
                    to_add = costs_df.loc[el].sum()
                sector_dic[el] = to_add*chf2eur
            cost_dic[sector] = sector_dic
        yearly_costs[year] = cost_dic
        
    trans_costs = {}
    for sector in yearly_costs[year].keys():
        for tech in yearly_costs[year][sector].keys():
            sector_costs = [yearly_costs[year][sector][tech] for year in years]
            trans_costs[tech] = sector_costs
            
    
    plot_df = pd.DataFrame(trans_costs)
    
    col_to_drop = [key for key in trans_costs.keys() if sum(trans_costs[key]) < 5]
    plot_df.drop(columns=col_to_drop, inplace=True)
    
    if norm:
        yearly_tot = plot_df[plot_df.columns].sum(axis=1) #Shares of partial system demand (norm wrt selected sectors sum)
        # yearly_tot = plot_df.sum(axis=1) #Shares of global system demand (norm wrt all sectors sum)
        for tech in plot_df.columns:
            plot_df[tech] = plot_df[tech]/yearly_tot
            
    pattern_map = {}
    patterns = ["-", "+", "x", "."]
    for i, sect in enumerate(sectors):
        for tech in used_sect[sect]:
            pattern_map[tech] = patterns[i%len(patterns)]
            
    to_melt = plot_df.copy(deep=True)
    to_melt['YEARS'] = years
    
    melted_df = pd.melt(to_melt, id_vars='YEARS', var_name='Tech', value_name='cost')
    
    colors = ["#006CB5", "#FFFF00","#00B050", "#996633", "#336600", "#FFE697", "#FFC000", "#808000", "#000000", "#FF9999", "#00B0F0", "#BD5BA6", "#3F47D0", "#A5A5A5", "#7F7F7F", "#7030A0", "rosybrown"]
    res = ['BIODIESEL', 'RES_SOLAR', 'RES_WIND', 'WOOD', 'WET_BIOMASS', 'GAS_RE', 'GAS', 'WASTE', 'COAL', 'URANIUM', 'ELECTRICITY', 'H2', 'AMMONIA_RE', 'DIESEL', 'GASOLINE', 'LFO', 'BIOETHANOL']
    res_color_dic = {res[i]: colors[i] for i in range(len(res))}
    
    if plot:
        if len(sectors) > 1:
            fig = px.area(melted_df, x='YEARS', y='cost', pattern_shape='Tech', pattern_shape_map=pattern_map, color='Tech', color_discrete_map=res_color_dic)
        else:
            fig = px.area(melted_df, x='YEARS', y='cost', color='Tech', color_discrete_map=res_color_dic)
    else:
        fig = None
    
    return plot_df, trans_costs, fig


# %% Work in progress

def primary_mix(path, years, plot=False):
    
    yearly_dic = {}
    
    for year in years:
        dir_name = '\YEAR_' + year
        
        res_bkdn = pd.read_table(path+dir_name+'\\resources_breakdown.txt')
        res_bkdn.drop_duplicates(inplace=True)
        res_cons = {res_bkdn.loc[idx]['Name']: res_bkdn.loc[idx]['Used'] for idx in res_bkdn.index}
        
        yearly_dic[year] = res_cons
    
    plot_dic = {}
    for res in yearly_dic[years[0]].keys():
        res_tot = [float(yearly_dic[year][res]) for year in years]
        plot_dic[res] = res_tot
    
    plot_df = pd.DataFrame(plot_dic)
    plot_df['YEARS'] = years
    
    if plot:
        fig = px.area(plot_df, x='YEARS', y=list(plot_dic.keys()))
    else:
        fig = None
        
    
    return yearly_dic, plot_df, fig

def sector_cons(path, years, sector_dic, plot=False):
    
    yearly_dic = {}
    
    for year in years:
        
        dir_name = '\YEAR_' + year
        year_bal = pd.read_table(path+dir_name+'\year_balance.txt', index_col='Tech')
        year_bal.drop(index=[' END_USES_DEMAND '], inplace=True)
        
        cons_dic = {}
        for sector in sector_dic.keys():
            for tech in year_bal.index:
                if year_bal.loc[tech][sector] > 0:
                    if tech in cons_dic.keys():
                        cons_dic[tech] += year_bal.loc[tech][sector]
                    else:
                        cons_dic[tech] = year_bal.loc[tech][sector]
        yearly_dic[year] = cons_dic
    
    plot_dic = {}
    for tech in yearly_dic[years[0]].keys():
        tech_var = [yearly_dic[year][tech] if tech in yearly_dic[year].keys() else 0 for year in years ]
        plot_dic[tech] = tech_var
    
    plot_df = pd.DataFrame(plot_dic)
    plot_df['YEARS'] = years
    
    if plot:
        fig = px.area(plot_df, x='YEARS', y=list(plot_dic.keys()))
    else:
        fig = None
    
    return yearly_dic, plot_dic, fig


#%% Case study costs plots

EUD_colors = {'HEAT_DEC': "#FF9999", 'HEAT_DHN': "firebrick", 'HEAT_HT': "#FF0000", 'ELECTRICITY': "#00B0F0", 'MOB_FREIGHT': "#996633", 'MOB_PUBLIC': "#FFC000", 'MOB_PRIVATE': "#BF9000"}

colors = ["#006CB5", "#FFFF00","#00B050", "#996633", "#336600", "#FFE697", "#FFC000", "#808000", "#000000", "#FF9999", "#00B0F0", "#BD5BA6", "#3F47D0", "#A5A5A5", "#7F7F7F", "#7030A0", "rosybrown"]
res = ['BIODIESEL', 'RES_SOLAR', 'RES_WIND', 'WOOD', 'WET_BIOMASS', 'GAS_RE', 'GAS', 'WASTE', 'COAL', 'URANIUM', 'ELECTRICITY', 'H2', 'AMMONIA_RE', 'DIESEL', 'GASOLINE', 'LFO', 'BIOETHANOL']
res_color_dic = {res[i]: colors[i] for i in range(len(res))}

wdth = 1520; hght = 715

sect = ['ELECTRICITY', 'HEAT_HT', 'HEAT_DHN', 'HEAT_DEC', 'MOB_PUBLIC', 'MOB_PRIVATE', 'MOB_FREIGHT']
res_sect = ['RENEW_RES', 'NON_RENEW_RES']

out = sector_costs(path, years, sectors=res_sect, plot=True, norm=True)
df_out = out[0]

melt_sect = []
for sect in df_out.columns:
    for el in df_out[sect]:
        if el > 0.1 and sect not in melt_sect:
            melt_sect.append(sect)
            
to_melt = df_out[melt_sect].copy(deep=True)
to_melt['YEARS'] = years
melt_plot = pd.melt(to_melt, id_vars='YEARS', var_name='Sector', value_name='Cost')

fig = px.line(melt_plot, x='YEARS', y='Cost', color='Sector', color_discrete_map=res_color_dic)

# fig = out[2]
# fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
fig.update_layout(legend_title_text='', 
                  xaxis=dict(title = 'Year', showgrid = False, zeroline = True),
                  yaxis=dict(title = 'Actualised yearly cost [M€_2020]', showgrid = False, zeroline = True),
                  plot_bgcolor='rgba(0,0,0,0)',
                  showlegend=False,
                  template="simple_white")
# fig.update_yaxes(range=(-0.01, 0.45))
# fig.write_image(graph_path+'\\main_res_costs_part.svg', width=wdth, height=hght)
fig.show()

# %% LSA plots
# LT_suffixes = ['base_case', '1_LT_renov_low', '6_LT_renov_high']
# heat_sectors = ['HEAT_DEC', 'HEAT_DHN']

    
# pass_mob_suffixes = ['base_case', '2_pass_mob_changes_low', '7_pass_mob_changes_high']
# pass_mob_sectors = ['MOB_PUBLIC', 'MOB_PRIVATE']


# freight_suffixes = ['base_case', '3_freight_changes_low', '8_freight_changes_high']
# freight_sectors = ['MOB_FREIGHT']

# pub_mob_suffixes = ['base_case', '4_mob_pub_max_2040', '5_mob_pub_max_2045']
# pub_mob_sectors = pass_mob_sectors

# suffixes = {'1_HEAT_LT': (LT_suffixes, heat_sectors), '2_PASS_MOB': (pass_mob_suffixes, pass_mob_sectors), '3_FREIGHT_MOB': (freight_suffixes, freight_sectors), '4_PUB_MOB': (pub_mob_suffixes, pub_mob_sectors)}


# norm_bool = True

# diff_df = pd.DataFrame()
# trans_cost_df = pd.DataFrame()
# tot_df = pd.DataFrame()

# for key in suffixes.keys():
#     suffs = suffixes[key][0]
#     sect = suffixes[key][1]
    
#     tot_costs = {}
    
#     to_plot_df = pd.DataFrame()
#     for suf in suffs:
#         if not os.path.exists(folder+'\graphs_'+key):
#             os.mkdir(folder+'\graphs_'+key)
            
#         if norm_bool:
#             dest_name = folder+'\graphs_'+key+'\\'+suf+'_norm.png'
#         else:
#             dest_name = folder+'\graphs_'+key+'\\'+suf+'.png'
            
#         out = sector_costs(folder+'\output_'+suf, years, sect, plot=True, norm=norm_bool)
        
#         df = sector_costs(folder+'\output_'+suf, years)[0]
        
#         tot_df[suf] = df['TOT_COST']
        
#         to_plot_df[suf] = tot_df[suf]
#         out = costs_per_sector(folder+'\output_'+suf, years, plot=True, norm=norm_bool)
        
        
        # fig = out[2]
        # fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
        # fig.update_layout(legend_title_text=suf, 
        #                   xaxis=dict(title = 'Year', showgrid = False, zeroline = True),
        #                   yaxis=dict(title = 'Actualised yearly cost [M€_2020]', showgrid = False, zeroline = True),
        #                   plot_bgcolor='rgba(0,0,0,0)',
        #                   showlegend=True,
        #                   template="simple_white")
        # # fig.write_image(dest_name, width=wdth, height=hght)
        # fig.show()
    
    # to_iter = [suf for suf in suffs if suf != 'base_case']
    
    # for suf in to_iter:
    #     diff_df[suf+'_BC_diff'] = 100*(abs(tot_df[suf] - tot_df['base_case']))/tot_df[suf]
    
    # to_plot_df['YEARS'] = years
        
    # melted_tot = pd.melt(to_plot_df, id_vars='YEARS', var_name='Case', value_name='Cost')
    # fig = px.line(melted_tot, x='YEARS', y='Cost', color='Case', color_discrete_sequence=['#000000', '#0000ff', '#ff0000'], line_dash='Case', line_dash_sequence=['solid', 'dot', 'dash'])
    # fig.update_layout(legend_title_text=suf, 
    #                   xaxis=dict(title = 'Year', showgrid = False, zeroline = True),
    #                   yaxis=dict(title = 'Actualised yearly cost [M€_2020]', showgrid = False, zeroline = True),
    #                   plot_bgcolor='rgba(0,0,0,0)',
    #                   showlegend=True,
    #                   template="simple_white")
    # # fig.update_yaxes(range=(0, 4.5e4), constrain='domain')
    # fig.show()
    # fig.write_image(folder+'\graphs_'+key+'\\yearly_costs_moche.svg', width=wdth, height=hght)
    
# Trans costs plot
# trans_cost = {}
# for case in tot_df.columns:
#     trans_cost[case] = np.trapz(y=tot_df[case], x=years.astype(float))

# x_plot = []
# y_plot = []
# color_seq = []

# for key in trans_cost.keys():
#     if key == 'base_case':
#         x_plot.append(key)
#         y_plot.append(trans_cost['base_case'])
#         color_seq.append('#000000')
#     else:
#         if 'low' in key.split('_') or '2040' in key.split('_'):
#             color_seq.append('#0000ff')
#         elif 'high' in key.split('_') or '2045' in key.split('_'):
#             color_seq.append('#ff0000')
#         x_plot.append(key)
#         y_plot.append(trans_cost[key] - trans_cost['base_case'])

# bar_df = pd.DataFrame()
# bar_df['case'] = x_plot
# bar_df['values'] = y_plot

# print(abs(bar_df.loc[0:]['values'])/bar_df.loc[0]['values'])
# print(np.mean(abs(bar_df.loc[1:]['values'])/bar_df.loc[0]['values']))

# fig = px.bar(bar_df, x='case', y='values', color='case', color_discrete_sequence=color_seq)
# fig.update_layout(legend_title_text=suf, 
#                   xaxis=dict(title = 'Year', showgrid = False, zeroline = True),
#                   yaxis=dict(title = 'Total transition costs diff [M€]', showgrid = False, zeroline = True),
#                   plot_bgcolor='rgba(0,0,0,0)',
#                   showlegend=True,
#                   template="simple_white")
# fig.update_yaxes(range=(-3e4, 10e4), constrain='domain')
# fig.show()
# fig.write_image(folder+'\graphs_LSA_general'+'\\total_trans_cost_diff_efuei.svg', width=wdth, height=hght)

# diff_df['YEARS'] = years
# melted_diff = pd.melt(diff_df, id_vars='YEARS', var_name='Case', value_name='Cost')

# fig = px.line(melted_diff, x='YEARS', y='Cost', color='Case')
# fig.update_layout(legend_title_text=suf, 
#                   xaxis=dict(title = 'Year', showgrid = False, zeroline = True),
#                   yaxis=dict(title = 'Normalized c_costs diff [%]', showgrid = False, zeroline = True),
#                   plot_bgcolor='rgba(0,0,0,0)',
#                   showlegend=True,
#                   template="simple_white")
# fig.show()