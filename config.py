project_root_path = 'D:\Pytorch Project\SSD'
train_data_name_path = 'dataset\pika'

HiXray = {
    # 'num_classes': 6,  !!!!
    'num_classes': 9,
    'lr_steps': (80000, 100000, 120000),
    # 'max_iter': 20000,
    'max_iter': 1000000,
    # 'feature_maps': [38, 19, 10, 5, 3],
    'feature_maps': [75, 38, 19, 10, 5],
    'min_dim': 300,
    'steps': [4, 8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_ratios': [[2], [2], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    # 'name': 'SIXray',
    'name': 'Xray20190723',
}

HiXray_CLASSES = (
    'Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
    'Nonmetallic_Lighter',
)
