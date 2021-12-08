import os
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import networkx as nx
import json
import util

interval = 2 # minutes
dataname = 'KL'
version = 'v1'
dataname = '%s_%s' % (dataname, version)

input_dir = 'D:/Gabby/OneDrive/WORK/COD/PRE/DataParse/hkgovDataAPI/data/irnAvgSpeed/%s'%version
outputdir = 'D:/Gabby/OneDrive/WORK/COD/PRE/Bigscity-LibCity/raw_data/%s/' % (dataname)
util.ensure_dir(outputdir)
outputdir_name = outputdir +  dataname

def load_json_as_dict(filename):
    with open(filename, 'r') as fp:
        data_dict = json.load(fp)    
    return data_dict


input_dir = 'D:\Data\Pre\Traffic\HK_Gov_road\cralwed_ss7049b\data/road_network\strategyRoadNetwork\generated/%s/' % dataname
strategy_CENTERLINE_df = pd.read_csv(input_dir + '/strategy_CENTERLINE_df.csv')

edge_weight1_dict = load_json_as_dict(input_dir + 'edge_weight1_dict.json')
ROUTE_idx_dict = load_json_as_dict(input_dir + 'ROUTE_idx_dict.json')
idx_ROUTE_dict = load_json_as_dict(input_dir + 'idx_ROUTE_dict.json')

# ROUTE_ID
strategy_CENTERLINE_df.ROUTE_ID = strategy_CENTERLINE_df.ROUTE_ID.astype(str)
strategy_CENTERLINE_df = strategy_CENTERLINE_df[strategy_CENTERLINE_df.ROUTE_ID.isin(ROUTE_idx_dict.keys())]
strategy_CENTERLINE_df.replace({'ROUTE_ID': ROUTE_idx_dict}, inplace=True)
features = ['ROUTE_ID', 'SHAPE_Length', 'ELEVATION', 'TRAVEL_DIRECTION']
rename_features = ['geo_id', 'length', 'elevation', 'direction']

geo = strategy_CENTERLINE_df[features]
geo['type'] = 'LineString'
geo.rename(columns=dict(zip(features, rename_features)), inplace=True)
geo.to_csv(outputdir_name+'.geo', index=False)



# DG = nx.read_gpickle(input_dir + 'strategy_roadnetwork_attr.gpkl')
# edgesView = list(DG.edges.data())
# edge_df = pd.DataFrame(edgesView, columns=['origin_id', 'destination_id', 'weight_dict'])
# edge_df['link_weight'] = edge_df['weight_dict'].apply(pd.Series, index=['weight'])

rel = []
rel_id = 0
all_pairs_dijkstra_path_length = load_json_as_dict(input_dir + 'all_pairs_dijkstra_path_length.json')
for source in all_pairs_dijkstra_path_length:
    for destination in all_pairs_dijkstra_path_length[source]:
        print([source, destination, all_pairs_dijkstra_path_length[source][destination]])
        rel.append([rel_id, 'geo', source, destination, all_pairs_dijkstra_path_length[source][destination]])
        rel_id += 1
rel_df = pd.DataFrame(rel, columns=['rel_id', 'type', 'origin_id', 'destination_id', 'dist']) 
# rel_df = edge_df[['origin_id', 'destination_id', 'link_weight']]
# rel_df['type'] = 'geo'
# rel_df['rel_id'] = range(rel_df.shape[0])
# rel_df = rel_df[['rel_id', 'type', 'origin_id', 'destination_id', 'link_weight']]
rel_df.to_csv(outputdir_name+'.rel', index=False)
print('rel_df shape: ', rel_df.shape)



fmt_df_dir = 'D:/Gabby/OneDrive/WORK/COD/PRE/DataParse/hkgovDataAPI/data/irnAvgSpeed/'
fmt_irnAvgSpeed_df_filename = fmt_df_dir + 'irnAvgSpeed_20210908_20210914.pkl.gz'
fmt_irnAvgSpeed_df1 = pd.read_pickle(fmt_irnAvgSpeed_df_filename, compression='gzip')
fmt_irnAvgSpeed_df_filename = fmt_df_dir + 'irnAvgSpeed_20210915_20210930.pkl.gz'
fmt_irnAvgSpeed_df2 = pd.read_pickle(fmt_irnAvgSpeed_df_filename, compression='gzip')
fmt_irnAvgSpeed_df_filename = fmt_df_dir + 'irnAvgSpeed_20211001_20211014.pkl.gz'
fmt_irnAvgSpeed_df3 = pd.read_pickle(fmt_irnAvgSpeed_df_filename, compression='gzip')
fmt_irnAvgSpeed_df_filename = fmt_df_dir + 'irnAvgSpeed_20211015_20211031.pkl.gz'
fmt_irnAvgSpeed_df4 = pd.read_pickle(fmt_irnAvgSpeed_df_filename, compression='gzip')
fmt_irnAvgSpeed_df_filename = fmt_df_dir + 'irnAvgSpeed_20211101_20211114.pkl.gz'
fmt_irnAvgSpeed_df5 = pd.read_pickle(fmt_irnAvgSpeed_df_filename, compression='gzip')

fmt_irnAvgSpeed_df = pd.concat([fmt_irnAvgSpeed_df1, fmt_irnAvgSpeed_df2, fmt_irnAvgSpeed_df3, fmt_irnAvgSpeed_df4, fmt_irnAvgSpeed_df5] )#, fmt_irnAvgSpeed_df3, fmt_irnAvgSpeed_df4], axis=0)
print(fmt_irnAvgSpeed_df.shape)


fmt_irnAvgSpeed_df.rename(columns=str, inplace=True)
fmt_irnAvgSpeed_df = fmt_irnAvgSpeed_df[list(ROUTE_idx_dict.keys())]
fmt_irnAvgSpeed_df.rename(columns=ROUTE_idx_dict, inplace=True)
# print('fmt_irnAvgSpeed_df cls 3: ', fmt_irnAvgSpeed_df.columns)
fmt_irnAvgSpeed_df.replace(r'', np.NaN, inplace=True)


segment_ids = fmt_irnAvgSpeed_df.columns

fmt_irnAvgSpeed_df = fmt_irnAvgSpeed_df[(fmt_irnAvgSpeed_df.index.hour > 5) & (fmt_irnAvgSpeed_df.index.hour < 23)][segment_ids]
print(fmt_irnAvgSpeed_df.shape)
# fmt_irnAvgSpeed_df.fillna( fmt_irnAvgSpeed_df.mean(skipna=True), inplace=True) #  [ts, num_links]
fmt_irnAvgSpeed_df.fillna(0, inplace=True) #  [ts, num_links]
# fmt_irnAvgSpeed_df.fillna(fmt_irnAvgSpeed_df.mean(skipna=True), inplace=True) #  [ts, num_links]



import seaborn as sns
import matplotlib.pyplot as plt
ts = fmt_irnAvgSpeed_df[fmt_irnAvgSpeed_df.columns[0]]
# ts = ts.cumsum()
ts.plot()
plt.show()

# cmap= sns.color_palette("coolwarm", as_cmap=True)
# sns.heatmap(values[...,4], cmap=cmap) #
# plt.show()




print('fmt_irnAvgSpeed_df shape: ', fmt_irnAvgSpeed_df.values.shape)
print('fmt_irnAvgSpeed_df mean: ', fmt_irnAvgSpeed_df.values.mean(axis=0).mean())
print('fmt_irnAvgSpeed_df std: ', fmt_irnAvgSpeed_df.values.std(axis=0).mean())
print('fmt_irnAvgSpeed_df std: ', fmt_irnAvgSpeed_df.values.std())





config = dict()
config['geo'] = dict()
config['geo']['including_types'] = ['LineString']
config['geo']['LineString'] = {
    'direction': 'enum',
    'length': 'num',
    'elevation': 'num'
}

config['rel'] = dict()
config['rel']['including_types'] = ['geo']
config['rel']['geo'] = {'link_weight': 'num'}
config['dyna'] = dict()
config['dyna']['including_types'] = ['state']
config['dyna']['state'] = {'entity_id': 'geo_id', 'traffic_speed': 'num'}
config['info'] = dict()
config['info']['data_col'] = 'traffic_speed'
config['info']['weight_col'] = 'dist'
config['info']['data_files'] = [dataname]
config['info']['geo_file'] = dataname
config['info']['rel_file'] = dataname
config['info']['output_dim'] = 1
config['info']['time_intervals'] = interval*60
config['info']['init_weight_inf_or_zero'] = 'inf'
config['info']['set_weight_link_or_dist'] = 'dist'
config['info']['calculate_weight_adj'] = 'true'
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(outputdir+'/config.json', 'w', encoding='utf-8'), ensure_ascii=True)



dyna_id = 0
dyna_file = open(outputdir_name+'.dyna', 'w')
dyna_file.write('dyna_id' + ',' + 'type' + ',' + 'time' + ',' + 'entity_id' + ',' + 'traffic_speed' + '\n')
for i in range(fmt_irnAvgSpeed_df.shape[0]):
    for j in range(fmt_irnAvgSpeed_df.shape[1]):
        time = fmt_irnAvgSpeed_df.index[i]
        entity_id = fmt_irnAvgSpeed_df.columns[j]
        traffic_speed = fmt_irnAvgSpeed_df.values[i, j]
        dyna_file.write(str(dyna_id) + ',' + 'state' + ',' + str(time)
                        + ',' + str(entity_id) + ',' + str(traffic_speed) + '\n')
        dyna_id += 1
dyna_file.close()

