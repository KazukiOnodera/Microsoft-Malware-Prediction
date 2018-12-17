"""

In [8]: y.value_counts(normalize=True)
Out[8]: 
90    0.294725
42    0.152013
65    0.125000
16    0.117737
15    0.063073
62    0.061672
88    0.047146
92    0.030454
67    0.026504
52    0.023318
95    0.022299
6     0.019241
64    0.012997
53    0.003823
Name: target, dtype: float64

for f in sorted(glob('*.py')):
#    print(f'nohup python -u {f} 0 > LOG/log_{f}.txt &')
    print(f'python -u {f} 0 > LOG/log_{f}.txt')

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from socket import gethostname
HOSTNAME = gethostname()

from tqdm import tqdm
#from itertools import combinations
from sklearn.model_selection import KFold
from time import time, sleep
from datetime import datetime
from multiprocessing import cpu_count, Pool
import gc

# =============================================================================
# global variables
# =============================================================================

COMPETITION_NAME = 'PLAsTiCC-2018'

SPLIT_SIZE = 100

TEST_LOGS = sorted(glob('../data/test_log*.pkl'))
AUG_LOGS = sorted(glob('../data/train_log_aug*.pkl'))

GENERATE_FEATURE_SIZE = 500

GENERATE_TEST = True

GENERATE_AUG = False

IMP_FILE = 'LOG/imp_801_cv.py-2.csv'

IMP_FILE_BEST = 'LOG/imp_used_934_predict_1120-1.py.csv'

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]

# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    send_line(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return

def reset_time():
    global st_time
    st_time = time()
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    send_line(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return

def elapsed_minute():
    return (time() - st_time)/60


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def to_feature(df, path):
    
    if df.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { df.columns[df.columns.duplicated()] }')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(f'{path}_{c}.f')
    return

def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    print(f'shape: {df.shape}')
    
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.p')
    return

def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print(f'reading {path}')
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df

#def to_feathers(df, path, split_size=3, inplace=True):
#    """
#    path = '../output/mydf'
#    
#    wirte '../output/mydf/0.f'
#          '../output/mydf/1.f'
#          '../output/mydf/2.f'
#    
#    """
#    if inplace==True:
#        df.reset_index(drop=True, inplace=True)
#    else:
#        df = df.reset_index(drop=True)
#    gc.collect()
#    mkdir_p(path)
#    
#    kf = KFold(n_splits=split_size)
#    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
#        df.iloc[val_index].to_feather(f'{path}/{i:03d}.f')
#    return
#
#def read_feathers(path, col=None):
#    if col is None:
#        df = pd.concat([pd.read_feather(f) for f in tqdm(sorted(glob(path+'/*')))])
#    else:
#        df = pd.concat([pd.read_feather(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
#    return df

def to_pkl_gzip(df, path):
    df.to_pickle(path)
    os.system('rm ' + path + '.gz')
    os.system('gzip ' + path)
    return
    
def save_test_features(df):
    for c in df.columns:
        df[[c]].to_pickle(f'../feature/test_{c}.pkl')
    return

# =============================================================================
# 
# =============================================================================
def get_dummies(df):
    """
    binary would be drop_first
    """
    col = df.select_dtypes('O').columns.tolist()
    nunique = df[col].nunique()
    col_binary = nunique[nunique==2].index.tolist()
    [col.remove(c) for c in col_binary]
    df = pd.get_dummies(df, columns=col)
    df = pd.get_dummies(df, columns=col_binary, drop_first=True)
    df.columns = [c.replace(' ', '-') for c in df.columns]
    return df


def reduce_mem_usage(df):
    col_int8 = []
    col_int16 = []
    col_int32 = []
    col_int64 = []
    col_float16 = []
    col_float32 = []
    col_float64 = []
    col_cat = []
    for c in tqdm(df.columns, mininterval=20):
        col_type = df[c].dtype

        if col_type != object:
            c_min = df[c].min()
            c_max = df[c].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    col_int8.append(c)
                    
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    col_int16.append(c)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    col_int32.append(c)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    col_int64.append(c)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    col_float16.append(c)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    col_float32.append(c)
                else:
                    col_float64.append(c)
        else:
            col_cat.append(c)
    
    if len(col_int8)>0:
        df[col_int8] = df[col_int8].astype(np.int8)
    if len(col_int16)>0:
        df[col_int16] = df[col_int16].astype(np.int16)
    if len(col_int32)>0:
        df[col_int32] = df[col_int32].astype(np.int32)
    if len(col_int64)>0:
        df[col_int64] = df[col_int64].astype(np.int64)
    if len(col_float16)>0:
        df[col_float16] = df[col_float16].astype(np.float16)
    if len(col_float32)>0:
        df[col_float32] = df[col_float32].astype(np.float32)
    if len(col_float64)>0:
        df[col_float64] = df[col_float64].astype(np.float64)
    if len(col_cat)>0:
        df[col_cat] = df[col_cat].astype('category')
        

def check_var(df, var_limit=0, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            df_ = df
#            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df
        
    var = df_.var()
    col_var0 = var[var<=var_limit].index
    if len(col_var0)>0:
        print(f'remove var<={var_limit}: {col_var0}')
    return col_var0

def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df
    
    corr = df_.corr('pearson').abs() # pearson or spearman
    a, b = np.where(corr>=corr_limit)
    col_corr1 = []
    for a_,b_ in zip(a, b):
        if a_ != b_ and a_ not in col_corr1:
#            print(a_, b_)
            col_corr1.append(b_)
    if len(col_corr1)>0:
        col_corr1 = df.iloc[:,col_corr1].columns
        print(f'remove corr>={corr_limit}: {col_corr1}')
    return col_corr1

def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var==False:
        col_corr1 = check_corr(df, corr_limit=corr_limit, sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return

def savefig_imp(imp, path, x='gain', y='feature', n=30, title='Importance'):
    import matplotlib as mpl
    mpl.use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(11.7, 8.27)
    sns.barplot(x=x, y=y, data=imp.head(n), label=x)
    plt.subplots_adjust(left=.4, right=.9)
    plt.title(title+' TOP{0}'.format(n), fontsize=20, alpha=0.8)
    plt.savefig(path)

# =============================================================================
# 
# =============================================================================

def load_train(col=None):
    if col is None:
        return pd.read_pickle('../data/train.pkl')
    else:
        return pd.read_pickle('../data/train.pkl')[col]

def load_test(col=None):
    if col is None:
        return pd.read_pickle('../data/test.pkl')
    else:
        return pd.read_pickle('../data/test.pkl')[col]

def load_target():
    return pd.read_pickle('../data/target.pkl')

def load_sub():
    return pd.read_pickle('../data/sub.pkl')

def check_feature():
    
    sw = False
    files = sorted(glob('../feature/train*.pkl'))
    for f in files:
        path = f.replace('train_', 'test_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    files = sorted(glob('../feature/test*.pkl'))
    for f in files:
        path = f.replace('test_', 'train_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    if sw:
        raise Exception('Miising file :(')
    else:
        print('All files exist :)')

def savefig_sub(sub, path):
    import matplotlib as mpl
    mpl.use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sub.iloc[:, 1:].hist(bins=50, figsize=(16, 12))
    plt.savefig(path)

def postprocess(sub:pd.DataFrame, weight=None, method='oli'):
    
    # fill 0
    oid_gal   = pd.read_pickle('../data/te_oid_gal.pkl').object_id
    oid_exgal = pd.read_pickle('../data/te_oid_exgal.pkl').object_id
    sub.loc[sub.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
    sub.loc[sub.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0
    
    # weight
    if weight is None:
        val = sub.iloc[:, 1:].values
        val = np.clip(a=val, a_min=0, a_max=1)
        weight = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
        weight = weight / val.sum(axis=0)
        print(f'weight = np.array({list(weight)})')
        val *= weight
        val /= val.sum(1)[:,None]
        sub.iloc[:, 1:] = val
    else:
        val = sub.iloc[:, 1:].values
        val = np.clip(a=val, a_min=0, a_max=1)
        val *= weight
        val /= val.sum(1)[:,None]
        sub.iloc[:, 1:] = val
        
    
    if method == 'giba':
        """
        Giba's postprocess
        """
        sub['class_99'] = 0
        sub.loc[sub.object_id.isin(oid_gal),   'class_99'] = 0.017
        sub.loc[sub.object_id.isin(oid_exgal), 'class_99'] = 0.17
        
    elif method == 'oli':
        preds_99 = np.ones(sub.shape[0])
        for i in range(1, 15): # should have oid
            preds_99 *= (1 - sub.iloc[:, i])
        sub['class_99'] = preds_99
        
    else:
        raise Exception(method)
        
    return

def plot_confusion_matrix(__file__, y_pred, normalize=True,
                          title='Confusion Matrix'):
    
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools
    import utils_metric
    
    classes = ['class_6',
             'class_15',
             'class_16',
             'class_42',
             'class_52',
             'class_53',
             'class_62',
             'class_64',
             'class_65',
             'class_67',
             'class_88',
             'class_90',
             'class_92',
             'class_95']
    
    y = load_target().target
    
    target_dict = {}
    target_dict_r = {}
    for i,e in enumerate(y.sort_values().unique()):
        target_dict[e] = i
        target_dict_r[i] = e
    
    y = y.replace(target_dict).values
    
    score = utils_metric.multi_weighted_logloss(y, y_pred)
    
    cnf_matrix = confusion_matrix(y, np.argmax(y_pred, axis=-1))
    np.set_printoptions(precision=2)
    
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)
    
    plt.figure(figsize=(12,12))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title}: {round(score, 5)}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'LOG/CM_{__file__}.png')
    
    send_line(f'Confusion Matrix wmlogloss: {score}', png=f'LOG/CM_{__file__}.png')
    
    return

# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API'):
    os.system(f'kaggle competitions submit -c {COMPETITION_NAME} -f {file_path} -m "{comment}"')
    sleep(60*5) # tekito~~~~
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'
    send_line(message.rstrip())

import requests
def send_line(message, png=None):
    
    line_notify_token = 'DUVuLOPCe26UrouWZrlIjzJd0zCiOFrVBGGtLEwFHV1'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    
    if png is None:
        requests.post(line_notify_api, data=payload, headers=headers)
    elif png is not None and png.endswith('.png'):
        files = {"imageFile": open(png, "rb")}
        requests.post(line_notify_api, data=payload, headers=headers, files=files)
    else:
        raise Exception('???', png)
    print(message)

def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
    os.system(f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')
    
    
