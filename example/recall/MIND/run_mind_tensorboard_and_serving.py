import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepmatch.utils import *
from deepmatch.models import *
from preprocess import gen_data_set, gen_model_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from keras.callbacks import TensorBoard

if __name__ == '__main__':
    data = pd.read_csv("../movielens_sample.txt")
    # 电影id 用户id 性别 年龄 职业 邮编
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]

    SEQ_LEN = 50

    # 1. Label Encoding for sparse features
    # 即稀疏特性数据特性化处理
    features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        # data[feature] each element 加一
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    debug_profile = data[["user_id", "gender", "age", "occupation", "zip"]]
    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates("movie_id")

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")["movie_id"].apply(list)

    train_set, test_set = gen_data_set(data, 0)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2. count unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 16

    user_feature_columns = [SparseFeat("user_id", feature_max_idx["user_id"], embedding_dim),
                            SparseFeat("gender", feature_max_idx["gender"], embedding_dim),
                            SparseFeat("age", feature_max_idx["age"], embedding_dim),
                            SparseFeat("occupation", feature_max_idx["occupation"], embedding_dim),
                            SparseFeat("zip", feature_max_idx["zip"], embedding_dim),
                            VarLenSparseFeat(SparseFeat("hist_movie_id", feature_max_idx["movie_id"], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # Define Model and Train
    # 设置学习状态, 类似TensorFlow global_initialization
    K.set_learning_phase(True)
    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        # 禁止使用eager_execution 运行, 区别于global_initialization
        tf.compat.v1.disable_eager_execution()

    model = MIND(user_feature_columns, item_feature_columns, dynamic_k=False, p=1, k_max=2, num_sampled=5,
                 user_dnn_hidden_units=(64, embedding_dim))

    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")

    keras_tb = TensorBoard(log_dir="../logs", histogram_freq=0, write_graph=True, write_images=False)
    print(model.summary())
    # tf.keras.utils.plot_model(model, to_file='MIND_model.png', show_shapes=True)

    history = model.fit(train_model_input, train_label,  # train_label
                        batch_size=256, epochs=1, verbose=1, validation_split=0.0, callbacks=[keras_tb])

    # model.save('mind_model')
    # model2 = tf.keras.models.load_model('mind_model.h5')

    # tf.keras.models.save_model(model, '../logs/pb')

    # 4. Generate user features for testing and full item features for retrieval(检索)
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile["movie_id"].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0, k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)

    # 5. [Optional] ANN search for faiss and evaluate the result
    # test_true_label = {line[0]: [line[2]] for line in test_set}
    #
    # import numpy as np
    # import faiss
    # from tqdm import tqdm
    # from deepmatch.utils import recall_N
    #
    # index = faiss.IndexFlatIP(embedding_dim)
    #
    # index.add(item_embs)
    # # 将一个内存不连续存储的数组转换为内存连续存储的数组
    # D, I = index.search(np.ascontiguousarray(user_embs), 50)
    # s = []
    # hit = 0
    # for i, uid in tqdm(enumerate(test_user_model_input["user_id"])):
    #     try:
    #         pred = [item_profile["movie_id"].values[x] for x in I[i]]
    #         filter_item = None
    #         recall_score = recall_N(test_true_label[uid], pred, N=50)
    #         s.append(recall_score)
    #         if test_true_label[uid] in pred:
    #             hit += 1
    #     except:
    #         print(i)
    #
    # print("recall", np.mean(s))
    # print("hr", hit / len(test_user_model_input["user_id"]))

    import tempfile

    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        user_embedding_model,
        os.path.join(export_path, 'user'),
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    # tf.keras.models.save_model(
    #     item_embedding_model,
    #     os.path.join(export_path, 'item'),
    #     overwrite=True,
    #     include_optimizer=True,
    #     save_format=None,
    #     signatures=None,
    #     options=None
    # )