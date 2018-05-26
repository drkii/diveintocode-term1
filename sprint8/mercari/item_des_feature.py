from sklearn.feature_extraction.text import TfidfVectorizer
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


def item_des_tfidf(df, length):
    tv = TfidfVectorizer(max_features=1000,
                         ngram_range=(1, 3),
                         stop_words='english')
    tv.fit(df['item_description'])

    X_train_description = tv.transform(df['item_description'][:length])
    X_test_description = tv.transform(df['item_description'][length:])

    return X_train_description, X_test_description


def get_X_description(train, test):
    '''
    商品説明欄のカウントベクトル化したsparse matrixを返す
    :param train: training dataset type :pandas.DataFrame
    :param test: test dataset type :pandas.DataFrame
    :return: sparse matrix
    '''
    len_train = len(train)
    concat_df = concat_train_test(train, test)
    train_description, test_description = item_des_tfidf(concat_df, len_train)

    return train_description, test_description
