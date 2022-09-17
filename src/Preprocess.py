import numpy as np
import string
import numpy as np
import pandas as pd
from sklearn import preprocessing


aymen_csv = pd.read_csv('resWords.csv')
aymen_csv["image_path"] = aymen_csv["IMAGE_ID"].apply(lambda x: f"Words/"+x)
upper = list(string.ascii_uppercase)
numbers = [i for i in range(10)]
aymen_csv["LABEL TYPE"] = aymen_csv["LABEL"].apply(lambda x: "phrase" if x[0] in upper else "numbers")
train_csv = aymen_csv
train_csv = train_csv[['IMAGE_ID', 'LABEL','image_path']]
train_csv = train_csv.reset_index(drop=True)
# Droping misslabeled images 
train_csv = train_csv[~train_csv['IMAGE_ID'].isin(['Amount in numbers_000007.jpg',
                              'Amount in numbers_000017.jpg', 
                              'Amount in numbers_000038.jpg',
                              'Amount in numbers_000043.jpg',
                              'Amount in numbers_000057.jpg',
                              'Amount in numbers_000330.jpg',
                              'Amount in numbers_000477.jpg',
                              'Amount in numbers_000590.jpg', 
                              'Amount in numbers_000654.jpg', 
                              'Amount in numbers_000738.jpg',
                              'Amount in numbers_000917.jpg',
                              'Amount in numbers_001238.jpg', 
                              'Amount in numbers_001274.jpg', 
                              'Amount in numbers_001348.jpg', 'Amount in numbers_001377.jpg', 'Amount in numbers_001439.jpg', 'Amount in numbers_001442.jpg', 'Amount in numbers_001616.jpg', 'Amount in numbers_001753.jpg', 'Amount in numbers_001804.jpg', 'Amount in numbers_001969.jpg', 'Amount in numbers_002103.jpg', 'Amount in numbers_002248.jpg', 'Amount in numbers_002253.jpg', 'Amount in numbers_002502.jpg', 'Amount in numbers_002523.jpg', 'Amount in numbers_002581.jpg', 'Amount in numbers_002639.jpg', 'Amount in numbers_002915.jpg', 'Amount in numbers_002951.jpg', 'Amount in numbers_003026.jpg', 'Amount in numbers_003059.jpg', 'Amount in numbers_003172.jpg', 'Amount in numbers_003258.jpg', 'Amount in numbers_003527.jpg', 'Amount in numbers_003570.jpg', 'Amount in numbers_003638.jpg', 'Amount in numbers_003700.jpg', 'Amount in numbers_003743.jpg', 'Amount in numbers_003927.jpg', 'Amount in numbers_003949.jpg', 'Amount in numbers_003957.jpg', 'Amount in numbers_004016.jpg', 'Amount in numbers_004045.jpg', 'Amount in numbers_004125.jpg', 'Amount in numbers_004144.jpg', 'Amount in numbers_004184.jpg', 'Amount in numbers_004196.jpg', 'Amount in numbers_004261.jpg', 'Amount in numbers_004315.jpg', 'Amount in numbers_004328.jpg', 'Amount in numbers_004366.jpg', 'Amount in numbers_004417.jpg', 'Amount in numbers_004477.jpg', 'Amount in numbers_004502.jpg', 'Amount in numbers_004622.jpg', 'Amount in numbers_004645.jpg', 'Amount in numbers_004679.jpg', 'Amount in numbers_004906.jpg', 'Amount in numbers_004993.jpg', 'Amount in numbers_005037.jpg', 'Amount in numbers_005190.jpg', 'Amount in numbers_005203.jpg', 'Amount in numbers_005247.jpg', 'Amount in numbers_005528.jpg', 'Amount in numbers_005573.jpg', 'Amount in numbers_005645.jpg', 'Amount in numbers_005696.jpg', 'Amount in numbers_005776.jpg', 'Amount in numbers_005799.jpg', 'Amount in numbers_005946.jpg', 'Amount in numbers_006073.jpg', 'Amount in numbers_006123.jpg', 'Amount in numbers_006158.jpg', 'Amount in numbers_006361.jpg', 'Amount in numbers_006406.jpg', 'Amount in numbers_006700.jpg', 'Amount in numbers_006951.jpg', 'Amount in numbers_006977.jpg'])]
train_csv = train_csv.reset_index(drop=True)
image_paths = train_csv['image_path']
image_files = train_csv['LABEL']
targets_orig = [x for x in image_files]
targets = [[c for c in x] for x in targets_orig]
targets_flat = [c for clist in targets for c in clist]
lbl_encoder = preprocessing.LabelEncoder()
lbl_encoder.fit(targets_flat)
targets_enc  = [lbl_encoder.transform(x) for x in targets]
targets_enc  = np.array(targets_enc ) 
train_csv["LABEL ENCODED"] = targets_enc
train_csv.to_csv('train_csv')
