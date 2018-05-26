from sklearn.feature_extraction.text import CountVectorizer
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


def split_cat(text):
    # textを'/'で区切る
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def item_des(df):
    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: split_cat(x)))
    return df


def to_categorical(df):
    df['general_cat'] = df['general_cat'].astype('category')
    df['subcat_1'] = df['subcat_1'].astype('category')
    df['subcat_2'] = df['subcat_2'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')
    return df


def CV(df, length):
    cv = CountVectorizer(min_df=5)
    cv.fit(df['general_cat'])
    X_train_category1 = cv.transform(df['general_cat'][:length])
    X_test_category1 = cv.transform(df['general_cat'][length:])

    return X_train_category1, X_test_category1


def get_main_category(train, test):
    len_train = len(train)
    concat_df = concat_train_test(train, test)
    concat_df = item_des(concat_df)
    concat_df = to_categorical(concat_df)
    train_cat1, test_cat1 = CV(concat_df, len_train)

    return train_cat1, test_cat1
