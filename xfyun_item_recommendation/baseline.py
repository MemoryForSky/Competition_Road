import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./训练集/train.txt', header=None)
test = pd.read_csv('./测试集/apply_new.txt', header=None)

train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']

data = pd.concat([train, test], axis=0)

for col in [x for x in data.columns if x not in ['label']]:
    data[col] = data[col].fillna(-1)
    data[col] = data[col].astype('str')

# 展开 tageid 和 time 信息
data['tagid'] = data['tagid'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
data['tagid'] = data['tagid'].apply(lambda x: x.split(','))

# 时间序列处理
data['time'] = data['time'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
data['time'] = data['time'].apply(lambda x: x.split(','))

pid = []
label = []
gender = []
age = []
tagid = []
time = []
province = []
city = []
model = []
make = []

for sub_data in data.values:
    s_tagid = sub_data[4]
    s_time = sub_data[5]
    for x, y in zip(s_tagid, s_time):
        pid.append(sub_data[0])
        label.append(sub_data[1])
        gender.append(sub_data[2])
        age.append(sub_data[3])
        tagid.append(x)
        time.append(y)
        province.append(sub_data[6])
        city.append(sub_data[7])
        model.append(sub_data[8])
        make.append(sub_data[9])
new_data = pd.DataFrame()

new_data['pid'] = pid
new_data['label'] = label
new_data['gender'] = gender
new_data['age'] = age
new_data['tagid'] = tagid
new_data['time'] = time
new_data['province'] = province
new_data['city'] = city
new_data['model'] = model
new_data['make'] = make

new_data['label'] = new_data['label'].fillna(-1)

new_data['time'] = new_data['time'].astype(float)
new_data['time'] = new_data['time'].astype('int64')

new_data['date'] = pd.to_datetime(new_data['time'], unit='ms')

new_data = new_data.sort_values(['pid', 'time'])

new_data_max_time = new_data.groupby(['pid'])['time'].max().reset_index()
new_data_max_time.columns = ['pid','max_time']
new_data = pd.merge(new_data,new_data_max_time,on=['pid'],how='left')
new_data['long'] = new_data['max_time'] - new_data['time']

# 计算用户的兴趣变化次数 统计用户兴趣出现几次变化
user_tagid_change = new_data.groupby(['pid'])['time'].nunique().reset_index()
user_tagid_change.rename(columns={'time': 'pid_tagid_change_time'}, inplace=True)
data = pd.merge(data, user_tagid_change, on=['pid'], how='left')

agg_list = {
    'time': ['mean', 'max', 'min', 'var', 'median','count'],
}

pid_time_feature = new_data.groupby(['pid']).agg(
    agg_list
)

pid_time_feature.columns = [x[0] + '_' + x[1] for x in pid_time_feature.columns]
pid_time_feature = pid_time_feature.reset_index()
data = pd.merge(data, pid_time_feature, on=['pid'], how='left')

new_data_feature = [
    'pid_tagid_change_time',
    'time_mean', 'time_max', 'time_min', 'time_var', 'time_median','time_count',
]


# 特征处理，对于 'province', 'city', 'model','make' 选择交集内的特征，其余特征置为 -1
for col in ['province', 'city', 'model','make']:
    in_set = set(train[col].unique()) & set(test[col].unique())
    data.loc[data[col].isin(list(in_set)),col] = data[col]
    data.loc[~data[col].isin(list(in_set)),col] = -1


# 二阶统计特征
us_feature = []
# count 编码 ，和lablending编码一个意思，一种编码方式
for col in ['gender', 'age', 'province', 'city', 'model','make']:
    data['{}_count'.format(col)] = data.groupby(col)[col].transform('count')
    data['{}_category'.format(col)] = data[col].astype('category')
    data['{}_category'.format(col)] = data['{}_category'.format(col)].cat.codes
    us_feature.append('{}_count'.format(col))
    us_feature.append('{}_category'.format(col))

corss_feature = ['gender', 'age', 'province', 'city', 'model','make']
# 交叉组合统计，就是组合特征的共现频次
while len(corss_feature) != 0:
    f = corss_feature.pop()
    for col in corss_feature:
        data['{}_{}_count'.format(f, col)] = data.groupby([f, col])[col].transform('count')
        data['{}_{}_category'.format(f, col)] = data[f] + '_' + data[col]
        data['{}_{}_category'.format(f, col)] = data['{}_{}_category'.format(f, col)].astype('category')
        data['{}_{}_category'.format(f, col)] = data['{}_{}_category'.format(f, col)].cat.codes
        us_feature.append('{}_{}_count'.format(f, col))
        us_feature.append('{}_{}_category'.format(f, col))

# 数据的特点是 训练集 的 label 是 五五开！
data['tagid'] = data['tagid'].apply(lambda x: ' '.join(x))

def emb(df, f1, f2):
    emb_size = 16
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, vector_size=emb_size, window=5, min_count=5, sg=0, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv.get_vector(w))
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    feature = []
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
        feature.append('{}_{}_emb_{}'.format(f1, f2, i))
    del model, emb_matrix, sentences
    return tmp, feature


emb_cols = [
    ['pid', 'tagid'],
]
for f1, f2 in emb_cols:
    tmp, f = emb(new_data, f1, f2)
    data = data.merge(tmp, on=f1, how='left')
    us_feature.extend(f)


# 按照时间信息，抽取对应的tagid和time信息
new_tagid_list = new_data.groupby(['pid'])['tagid'].apply(list).reset_index()
new_tagid_list.columns = ['pid','new_tagid']

data = pd.merge(data,new_tagid_list,on=['pid'],how='left')

def get_list_info(x,i):
    if len(x) <= (-i):
        x = x + [-1] * (-i - len(x))
    return int(x[i])

for i in range(1,11):
    data['net_tagid_{}'.format(i)] = data['new_tagid'].apply(lambda x:get_list_info(x,-i))
    us_feature.append('net_tagid_{}'.format(i))


us_feature = us_feature + new_data_feature

target_list = ['age', 'province', 'city', 'model']

# 特征unique count特征
for index, col1 in enumerate(['age', 'province', 'city', 'model']):
    for col2 in ['age', 'province', 'city', 'model'][index:]:
        data['{}_in_{}_count'.format(col1, col2)] = data.groupby(col1)[col2].transform('count')
        data['{}_in_{}_nunique'.format(col1, col2)] = data.groupby(col1)[col2].transform('nunique')
        data['{}_in_{}_nunique/{}_in_{}_count'.format(col1, col2, col1, col2)] = data['{}_in_{}_nunique'.format(col1,
                                                                                                                col2)] / \
                                                                                 data['{}_in_{}_count'.format(col1,
                                                                                                              col2)]

        data['{}_in_{}_count'.format(col2, col1)] = data.groupby(col2)[col1].transform('count')
        data['{}_in_{}_nunique'.format(col2, col1)] = data.groupby(col2)[col1].transform('nunique')
        data['{}_in_{}_nunique/{}_in_{}_count'.format(col2, col1, col2, col1)] = data['{}_in_{}_nunique'.format(col2,
                                                                                                                col1)] / \
                                                                                 data['{}_in_{}_count'.format(col2,
                                                                                                              col1)]

        us_feature.append('{}_in_{}_count'.format(col1, col2))
        us_feature.append('{}_in_{}_nunique'.format(col1, col2))
        us_feature.append('{}_in_{}_nunique/{}_in_{}_count'.format(col1, col2, col1, col2))

        us_feature.append('{}_in_{}_count'.format(col2, col1))
        us_feature.append('{}_in_{}_nunique'.format(col2, col1))
        us_feature.append('{}_in_{}_nunique/{}_in_{}_count'.format(col2, col1, col2, col1))

train = data[:train.shape[0]]
test = data[train.shape[0]:]

# 引入kflod转化率
skf_ratdio = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_train_ratdio = np.zeros((train.shape[0], len(target_list)))
oof_test_ratdio = np.zeros((test.shape[0], len(target_list)))

for index, (train_index, valid_index) in enumerate(skf_ratdio.split(train, train['label'])):
    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
    X_test = test.copy()
    for k, col in enumerate(target_list):
        t = X_train.groupby(col)[['label']].mean().to_dict()
        X_valid['ratdio_{}'.format(col)] = X_valid[col].map(t['label'])
        oof_train_ratdio[valid_index, k] = X_valid['ratdio_{}'.format(col)].values
        X_test['ratdio_{}'.format(col)] = X_test[col].map(t['label'])
        oof_test_ratdio[:, k] = oof_test_ratdio[:, k] + X_test['ratdio_{}'.format(col)].values / skf_ratdio.n_splits

for i in range(0, len(target_list)):
    train[i] = oof_train_ratdio[:, i]
    test[i] = oof_test_ratdio[:, i]
    us_feature.append(i)

us_feature.append('gender')
us_feature.append('age')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_train = np.zeros(shape=(train.shape[0]))
oof_test = np.zeros(shape=(test.shape[0]))

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'verbose': 0,
    'random_state': 42,
    'n_jobs': -1,
}
imp_Df = pd.DataFrame()
imp_Df['feature'] = us_feature

for index, (train_index, valid_index) in enumerate(skf.split(train, train['label'])):
    X_train, X_valid = train.iloc[train_index][us_feature].values, train.iloc[valid_index][us_feature].values
    y_train, y_valid = train.iloc[train_index]['label'], train.iloc[valid_index]['label']
    print(index)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_valid, label=y_valid)
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dval],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    X_valid_pred = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    imp_Df[index] = lgb_model.feature_importance()

    oof_train[valid_index] = X_valid_pred
    oof_test = oof_test + lgb_model.predict(test[us_feature].values,
                                            num_iteration=lgb_model.best_iteration) / skf.n_splits

train['predict'] = oof_train
train['rank'] = train['predict'].rank()
train['p'] = 1
train.loc[train['rank'] <= train.shape[0] * 0.5, 'p'] = 0
# train['pre'] = train['predict'].apply(lambda x:1 if x>0.5 else 0)

bst_f1_tmp = f1_score(train['label'].values, train['p'].values)
print(bst_f1_tmp)
# bst_f1_tmp = f1_score(train['label'].values, train['pre'].values)
# print(bst_f1_tmp)
# bst_f1 = 0
# bst_thr = 0
# for thr in range(0,11):
#     thr = thr / 10
#     tmp = np.where(oof_train > thr,1,0)
#     bst_f1_tmp = f1_score(train['label'].values,tmp)
#     if bst_f1 < bst_f1_tmp:
#         bst_f1 = bst_f1_tmp
#         bst_thr = thr

# print(bst_f1,bst_thr)
#
submit = test[['pid']]
submit['tmp'] = oof_test
submit.columns = ['user_id', 'tmp']

submit['rank'] = submit['tmp'].rank()
submit['category_id'] = 1
submit.loc[submit['rank'] <= int(submit.shape[0] * 0.5), 'category_id'] = 0
# submit['pre'] = submit['tmp'].apply(lambda x:1 if x>0.5 else 0)

print(submit['category_id'].mean())

submit[['user_id', 'category_id']].to_csv('open_{}.csv'.format(str(bst_f1_tmp).split('.')[1]), index=False)
# print(submit['pre'].mean())