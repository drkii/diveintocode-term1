from sklearn.preprocessing import LabelBinarizer
import pandas as pd


def concat_train_test(train, test):
    # trainとtestのidカラム名を変更する
    train = train.rename(columns={'train_id': 'id'})
    test = test.rename(columns={'test_id': 'id'})

    # 両方のセットへ「is_train」のカラムを追加
    # 1 = trainのデータ、0 = testデータ
    train['is_train'] = 1
    test['is_train'] = 0

    # trainのprice(価格）以外のデータをtestと連結
    train_test_combine = pd.concat([train.drop(['price'], axis=1), test], axis=0)
    return train_test_combine


def brand_labeling(df, length):
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(df['brand_name'])
    #   X_brand = lb.fit_transform(df['brand_name'])
    X_train_brand = lb.transform(df['brand_name'][:length])
    X_test_brand = lb.transform(df['brand_name'][length:])

    return X_train_brand, X_test_brand


def get_brand_labeling(train, test):
    len_train = len(train)
    concat_df = concat_train_test(train, test)

    X_train_brand, X_test_brand = brand_labeling(concat_df, len_train)

    return X_train_brand, X_test_brand
