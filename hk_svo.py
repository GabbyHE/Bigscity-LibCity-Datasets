# link: https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS08
import numpy as np
import pandas as pd
import json
import util
import os

interval = 4
part_detector_id = 'AID011'

input_dir = 'D:/Gabby/OneDrive/WORK/COD/PRE/DataParse/hkgovDataAPI/data/SVO/'
outputdir = 'D:/Gabby/OneDrive/WORK/COD/PRE/Bigscity-LibCity/raw_data/'
util.ensure_dir(outputdir)

outputdir = outputdir+'%s/interval%s/' % (part_detector_id, interval)
util.ensure_dir(outputdir)
outputdir_name = outputdir +  part_detector_id 


# load traffic data
tp_dir_file = input_dir + '/fmt_agg_mydefined_ts/interval%s_speed_20211001_20211031.pkl.gz' % interval
speed_df = pd.read_pickle(tp_dir_file, compression='gzip')
part_cls = speed_df.columns[speed_df.columns.str.contains(part_detector_id)]
part_speed_df = speed_df[part_cls]
speed = part_speed_df.values
np.save(outputdir_name+'_speed', speed)
print('speed', speed.shape)
detector_lane_ids = part_speed_df.columns

geo_file = outputdir_name + '.geo'
if os.path.isfile(geo_file):
    geo=pd.read_csv(geo_file)
else:
    # detector_lane_info
    detector_traffic_speed_volume_occ_df = pd.read_csv(input_dir+'detector_traffic_speed_volume_occ_info.csv')
    
    detector_lane_id_df = pd.read_csv(input_dir+'detector_lane_id_df.csv')
    detector_lane_id_df1 = detector_lane_id_df.copy()
    detector_lane_id_df1 = detector_lane_id_df[detector_lane_id_df.detector_lane_id.isin(detector_lane_ids)]
    detector_lane_id_df1['geo_id'] = detector_lane_id_df1['detector_lane_id']
    # detector_lane_id_df1 = detector_lane_id_df1[detector_lane_id_df1['geo_id'].str.contains('%s' % part_detector_id)]
    detector_lane_id_df1.sort_values('geo_id',inplace=True)
    print('detector_lane_id_df1 1', detector_lane_id_df1.shape)
    detector_lane_id_df1[['AID_ID_Number', 'lane_id']] = detector_lane_id_df1['detector_lane_id'].str.split('-', expand=True)
    print('detector_lane_id_df1 2', detector_lane_id_df1.shape)


    detector_lane_df = detector_lane_id_df1.merge(detector_traffic_speed_volume_occ_df, how='left', on='AID_ID_Number')
    detector_lane_df['coordinates'] = detector_lane_df['Latitude'].astype(str)  +',' + detector_lane_df['Longitude'].astype(str) 
    detector_lane_df['type'] = 'Point'

    geo = detector_lane_df[['geo_id', 'type', 'coordinates']]
    geo.to_csv(outputdir_name + '.geo', index=False)
    # geo = geo[geo['geo_id'].str.contains('%s' % part_detector_id)]
    print('geo_df', geo.shape)


detector_lane_edge_df = pd.read_csv(input_dir + '/Bigcity/'+'%s_detector.rel' % part_detector_id, index_col=False)
old_cls = detector_lane_edge_df.columns
print(old_cls)
columns = ['origin_id', 'destination_id', 'distance', 'cost', 'rel_id', 'type']
cls_dict = dict(zip(old_cls, columns))
detector_lane_edge_df.rename(columns=cls_dict, inplace=True)
print('detector_lane_edge_df', detector_lane_edge_df.columns)


rel_df = detector_lane_edge_df[['rel_id', 'type', 'origin_id', 'destination_id', 'cost']]
rel_df.to_csv(outputdir_name +'.rel', index=False)
# print('rel_df', rel_df['cost'])
# dataset = np.stack([dataset_spd[list(geo.geo_id)].values, dataset_vol[list(geo.geo_id)].values, dataset_occ[list(geo.geo_id)].values], axis=2)


# nuild timeslot list
start_time = util.datetime_timestamp('2021-10-01T00:00:00Z')
end_time = util.datetime_timestamp('2021-10-31T23:59:59Z')
# end_time = util.datetime_timestamp('2021-11-17T23:59:30Z')
timeslot = []
while start_time < end_time:
    timeslot.append(util.timestamp_datetime(start_time))
    start_time = start_time + interval*30
print('len timeslots', len(timeslot))

# generated .dyna file
dyna_id_counter = 0
dyna = []
geo_id_list = list(geo.geo_id)
print('geo_id_list', len(geo_id_list))
for j in range(speed.shape[1]):
    for i in range(speed.shape[0]):
        # time = num2time(i)
        time = timeslot[i]
        #               dyna_id,      type,    time, entity_id,     traffic_speed
        dyna.append([dyna_id_counter, 'state', time, geo_id_list[j], speed[i, j]])
        dyna_id_counter += 1
dyna = pd.DataFrame(dyna, columns=['dyna_id', 'type', 'time', 'entity_id', 'traffic_speed'])
dyna.to_csv(outputdir_name + '_speed.dyna', index=False)

# --------------------------Create .json table---------------------------------------------
config = dict()
config['geo'] = dict()
config['geo']['including_types'] = ['Point']
config['geo']['Point'] = {}
config['rel'] = dict()
config['rel']['including_types'] = ['geo']
config['rel']['geo'] = {'cost': 'num'}
config['dyna'] = dict()
config['dyna']['including_types'] = ['state']
config['dyna']['state'] = {'entity_id': 'geo_id', 'traffic_speed': 'num'}
config['info'] = dict()
config['info']['data_col'] = ['traffic_speed']
config['info']['weight_col'] = 'cost'
config['info']['data_files'] = [part_detector_id]
config['info']['geo_file'] = part_detector_id
config['info']['rel_file'] = part_detector_id
config['info']['output_dim'] = 1
config['info']['time_intervals'] = 300
config['info']['init_weight_inf_or_zero'] = 'inf'
config['info']['set_weight_link_or_dist'] = 'dist'
config['info']['calculate_weight_adj'] = False
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(outputdir + '/config.json', 'w', encoding='utf-8'), ensure_ascii=False)






"""
# dyna = []
dyna_id = 0
print(dataname+'.dyna')
dyna_file = open(dataname+'/AID042.dyna', 'w')
dyna_file.write('dyna_id' + ',' + 'type' + ',' + 'time' + ',' + 'entity_id'
                + ',' + 'traffic_flow' + ',' + 'traffic_occupancy' + ',' + 'traffic_speed' + '/n')
for j in range(dataset.shape[1]):
    entity_id = j  # 这个数据集的id是0-306
    for i in range(len(timeslot)):
        time = timeslot[i]
        # dyna.append([dyna_id, 'state', time, entity_id, dataset[i][j][0], dataset[i][j][1], dataset[i][j][2]])
        dyna_file.write(str(dyna_id) + ',' + 'state' + ',' + str(time)
                        + ',' + str(entity_id) + ',' + str(dataset[i][j][0])
                        + ',' + str(dataset[i][j][1]) + ',' + str(dataset[i][j][2]) + '/n')
        dyna_id = dyna_id + 1
        # if dyna_id % 10000 == 0:
        #     print(str(dyna_id) + '/' + str(dataset.shape[0]*dataset.shape[1]))
dyna_file.close()
# dyna = pd.DataFrame(dyna, columns=['dyna_id', 'type', 'time', 'entity_id', 'traffic_flow'])
# dyna.to_csv(dataname+'.dyna', index=False)


config = dict()
config['geo'] = dict()
config['geo']['including_types'] = ['Point']
config['geo']['Point'] = {}
config['rel'] = dict()
config['rel']['including_types'] = ['geo']
config['rel']['geo'] = {'cost': 'num'}
config['dyna'] = dict()
config['dyna']['including_types'] = ['state']
config['dyna']['state'] = {'entity_id': 'geo_id', 'traffic_flow': 'num',
                           'traffic_occupancy': 'num', 'traffic_speed': 'num'}
config['info'] = dict()
config['info']['data_col'] = ['traffic_flow', 'traffic_occupancy', 'traffic_speed']
config['info']['weight_col'] = 'cost'
config['info']['data_files'] = ['AID042']
config['info']['geo_file'] = 'AID042'
config['info']['rel_file'] = 'AID042'
config['info']['output_dim'] = 3
config['info']['time_intervals'] = 300
config['info']['init_weight_inf_or_zero'] = 'zero'
config['info']['set_weight_link_or_dist'] = 'link'
config['info']['calculate_weight_adj'] = False
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(outputdir+'/config.json', 'w', encoding='utf-8'), ensure_ascii=False)
"""
