import numpy as np
import pandas as pd
import warnings
from scipy.sparse import csr_matrix, hstack, vstack
import mercari.create_count_feature as f1
import mercari.get_main_category as f2
import mercari.brand_labeling as f3
import mercari.item_des_feature as f4
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

path = '../input/'
train_df = pd.read_csv(path + 'train.tsv', sep='\t', encoding='utf-8', nrows=1000)
test_df = pd.read_csv(path + 'test.tsv', sep='\t', encoding='utf-8', nrows=1000)

############
##前処理
############

# 欠損値埋める
train_df['item_condition_id'].fillna(2, inplace=True)
test_df['item_condition_id'].fillna(2, inplace=True)
train_df['shipping'].fillna(0, inplace=True)
test_df['shipping'].fillna(0, inplace=True)
train_df['brand_name'].fillna(value='missing', inplace=True)
test_df['brand_name'].fillna(value='missing', inplace=True)

# 説明欄の補正
train_df['item_description'].fillna('', inplace=True)
train_df['item_description'] = train_df['item_description'].replace('No description yet', '')
test_df['item_description'].fillna('', inplace=True)
test_df['item_description'] = test_df['item_description'].replace('No description yet', '')

############
# 特徴量追加
############

train_df, test_df = f1.create_count_features(train_df), f1.create_count_features(test_df)
train_cat1, test_cat1 = f2.get_main_category(train_df, test_df)
X_train_brand, X_test_brand = f3.get_brand_labeling(train_df, test_df)
train_description, test_description = f4.get_X_description(train_df, test_df)

use_cols = ['item_condition_id', 'shipping', 'num_words_item_description']
X_train = train_df[use_cols]
y_train = train_df['price'].values
X_test = test_df[use_cols]

# sparse_matrix化 カウントベクトル系の特徴量を追加
X_train = hstack((X_train, train_cat1, train_description, X_train_brand)).tocsr()
X_test = hstack((X_test, test_cat1, test_description, X_test_brand)).tocsr()


############
# 評価関数作成
#############
def root_mean_squared_log_error(truth, pred):
    return np.sqrt(mean_squared_log_error(truth, pred))


rmsle = make_scorer(root_mean_squared_log_error, greater_is_better=False)

######################
# STEP1 : 2モデルでの学習
######################

score_ridge_list = []
score_xgb_list = []

predict = np.zeros((y_train.shape[0], 2))

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X_train):
    cv_X_train, cv_X_test = X_train[train_index], X_train[test_index]
    cv_y_train, cv_y_test = y_train[train_index], y_train[test_index]

    # Ridge回帰
    model_ridge = Ridge(alpha=20)
    model_ridge.fit(cv_X_train, cv_y_train)
    y_predict_ridge = model_ridge.predict(cv_X_test)

    # xgboostモデルの作成
    reg = xgb.XGBRegressor()

    # ハイパーパラメータ探索
    reg_cv = GridSearchCV(reg, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
    reg_cv.fit(cv_X_train, cv_y_train)
    print(reg_cv.best_params_, reg_cv.best_score_)

    # 改めて最適パラメータで学習
    model_xgb = xgb.XGBRegressor(**reg_cv.best_params_)
    model_xgb.fit(cv_X_train, cv_y_train)
    y_predict_xgb = model_xgb.predict(cv_X_test)

    score_ridge = root_mean_squared_log_error(cv_y_test, y_predict_ridge)
    score_xgb = root_mean_squared_log_error(cv_y_test, y_predict_xgb)

    score_ridge_list.append(score_ridge)
    score_xgb_list.append(score_xgb)

    print("Ridge回帰：{}".format(score_ridge))
    print("XGBoost：{}".format(score_xgb))

    predict[test_index, 0] = y_predict_ridge
    predict[test_index, 1] = y_predict_xgb

######################
# STEP2 : 予測値を説明変数とした学習でスコアを検証
######################

score_stacking_list = []

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(predict):
    cv_X_train, cv_X_test = predict[train_index, :], predict[test_index, :]
    cv_y_train, cv_y_test = y_train[train_index], y_train[test_index]

    # xgboostモデルの作成
    reg = xgb.XGBRegressor()

    # ハイパーパラメータ探索
    reg_cv = GridSearchCV(reg, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
    reg_cv.fit(X_train, y_train)
    print(reg_cv.best_params_, reg_cv.best_score_)

    # 改めて最適パラメータで学習
    model_xgb = xgb.XGBRegressor(**reg_cv.best_params_)
    model_xgb.fit(cv_X_train, cv_y_train)
    y_predict_stacking = model_xgb.predict(cv_X_test)

    score = root_mean_squared_log_error(cv_y_test, y_predict_stacking)
    print("スタッキング：{}".format(score))
    score_stacking_list.append(score)

######################
# STEP3 : スコアを確認する
######################

print("RIDGE      score:", np.average(score_ridge_list))
print("XGB         score:", np.average(score_xgb_list))
print("Staking   score:", np.average(score_stacking_list))

######################
# スタッキングで学習
######################

# Ridge回帰
model_ridge = Ridge(alpha=20)
model_ridge.fit(X_train, y_train)
y_predict_ridge = model_ridge.predict(X_test)

# xgboostモデルの作成
reg = xgb.XGBRegressor()

# ハイパーパラメータ探索
# reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
# reg_cv.fit(cv_X_train, cv_y_train)
# print(reg_cv.best_params_, reg_cv.best_score_)

# XGBoostモデルで学習
model_xgb = xgb.XGBRegressor(**reg_cv.best_params_)
model_xgb.fit(X_train, y_train)
y_predict_xgb = model_xgb.predict(X_test)

# テストデータの予測値を説明変数とする
y_predict_two_model = np.array([y_predict_ridge, y_predict_xgb]).T

model_xgb = xgb.XGBRegressor(**reg_cv.best_params_)
model_xgb.fit(predict, y_train)
y_predict_stacking = model_xgb.predict(y_predict_two_model)

###################
# 提出ファイルの作成
###################
mysubmission = pd.DataFrame()
mysubmission['test_id'] = test_df['test_id']
mysubmission['price'] = y_predict_stacking
mysubmission.to_csv('ensemble_from_pyfile.csv', index=False)
