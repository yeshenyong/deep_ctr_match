import random
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data["movie_id"].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby("user_id")):
        # unique user movie_id list
        pos_list = hist["movie_id"].tolist()
        rating_list = hist["rating"].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # 每个reviewerID(每个userID) 的训练数据
                # 1 = {tuple: 6}(1, [112, 186], 84, 1, 2, 5)
                # 0 = {tuple: 6}(1, [186], 112, 1, 1, 4)
                # 2 = {tuple: 6}(1, [84, 112, 186], 52, 1, 3, 5)
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for neg in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + neg], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    # maybe wasting time part
    # {tuple: 6}(1, [84, 112, 186], 52, 1, 3, 5)
    train_uid = np.array([line[0] for line in train_set])

    # question 1: 为什么不同序列信息
    # question 2: 为什么train_label 为 1, "movie_id" -> "embedding" 为什么没有利用特征信息genres 呢?

    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # padding 后缀补0
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    # 补充train_model_input 的输入属性
    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input["user_id"]][key].values

    return train_model_input, train_label















