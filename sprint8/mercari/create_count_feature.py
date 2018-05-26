import numpy as np


def create_count_features(df_data):
    '''
    商品説明欄の単語数をカウントする
    :param df_data: pandas DataFrame
    :return:
    '''

    def lg(text):
        text = [x for x in text.split() if len(x) >= 3]
        return len(text)

    df_data['num_words_item_description'] = df_data['item_description'].apply(lg).astype(np.uint16)
    return df_data
