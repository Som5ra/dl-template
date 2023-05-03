# training
weight_saving_dir = "/home/risksis/highway_pcd_image_test/road_detection/weights/{config_name}/"
max_epoch = 200
batch_size = 10
ds_num_workers = 4
ds_prefetch_factor = 4

save_freq = 20
fp16 = False 

pretrained_model_path = None
gradient_accum = 1

loss_weights = dict()
loss = None