from action_predict import action_prediction
from pie_data import PIE
from jaad_data import JAAD
import os
import sys
import yaml
import time
import tensorflow as tf
import csv
# this works on tensorflow 2.8, windows 10, jupyterlab Version 3.3.2
# this is the very FIRST lines of code
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=46080)])
  #  tf.config.experimental.set_virtual_device_configuration(
   #     gpu,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    #)


#tf.config.set_visible_devices([], 'GPU')

# if tf.test.gpu_device_name(): # this lies and tells you about all devices
gpus1 = tf.config.list_logical_devices('GPU')
if tf.config.experimental.list_logical_devices('GPU'):
    print('GPU found')
else:
    print("No GPU found")
strategy = tf.distribute.MirroredStrategy(gpus1)
#@profile
def test_model(saved_files_path=None):
    #with strategy.scope():
    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)
    print(opts)
    model_opts = opts['model_opts']
    data_opts = opts['data_opts']
    net_opts = opts['net_opts']

    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else \
            model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    if model_opts['dataset'] == 'pie':
        pass
            # imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
            # imdb.get_data_stats()
    elif model_opts['dataset'] == 'jaad':
             imdb = JAAD(data_path='./Ahmed_Elgazwy/Pedestrian_Crossing_Intention_Prediction-main/JAAD')
             
    else:
        raise ValueError("{} dataset is incorrect".format(model_opts['dataset']))

    method_class = action_prediction(model_opts['model'])(**net_opts)
    #beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    #saved_files_path = method_class.train(beh_seq_train, **train_opts, model_opts=model_opts)
    #with strategy.scope():
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    
    
    with open("./Ahmed_Elgazwy/Pedestrian_Crossing_Intention_Prediction-main/test4.csv", "w") as outfile:
   
   # pass the csv file to csv.writer.
       writer = csv.writer(outfile)
     
    # convert the dictionary keys to a list
       key_list = list(beh_seq_test.keys())
       print(key_list)
       print(beh_seq_test['image_dimension'])
       print(beh_seq_test['image_dimension'])
    # find the length of the key_list
       limit = len(key_list)
     
    # the length of the keys corresponds to
    # no. of. columns.
       writer.writerow(beh_seq_test.keys())
     
    # iterate each column and assign the
    # corresponding values to the column
       for x in key_list:
           writer.writerow([beh_seq_test[x][0]])
    with open("./Ahmed_Elgazwy/Pedestrian_Crossing_Intention_Prediction-main/test5.csv", "w") as outfile:
   
   # pass the csv file to csv.writer.
       writer = csv.writer(outfile)
     
    # convert the dictionary keys to a list
       #key_list = list(beh_seq_test1.keys())
       #print(key_list)
       #print(beh_seq_test2.keys())
       #print(beh_seq_test2['vehicle_act'][0])
       #print(beh_seq_test1['image_dimension'])
       #print(beh_seq_test2['intent'])
       
    # find the length of the key_list
       limit = len(key_list)
     
    # the length of the keys corresponds to
    # no. of. columns.
       #writer.writerow(beh_seq_test1.keys())
     
    # iterate each column and assign the
    # corresponding values to the column
       #for x in key_list:
           #writer.writerow([beh_seq_test1[x][0]])
    test_results,true_labels = method_class.test(beh_seq_test, saved_files_path)
    
    print('test done')
    print(test_results, true_labels)


if __name__ == '__main__':
    with tf.device('/GPU:1'):
    #with tf.device('/GPU:0'):
    #with strategy.scope():
        saved_files_path = sys.argv[1]
        test_model(saved_files_path=saved_files_path)

