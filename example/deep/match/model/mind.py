"""
Author:
    Qingliang Cai,leocaicoder@163.com
    Weichen Shen,wcshen1994@164.com
Reference:
Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]//Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019: 2615-2623.
"""

"""
Researcher:
    Shenyong Ye 1050575224@qq.com
"""

from example.deep.feature_column import SparseFeat, build_input_features, VarLenSpareFeat, DenseFeat, \
    embedding_lookup, get_dense_input
from example.deep.inputs import create_embedding_matrix, varlen_embedding_lookup, get_varlen_pooling_list


def MIND(user_feature_columns, item_feature_columns, num_sampled=5, k_max=2, p=1.0, dynamic_k=False,
         user_dnn_hidden_units=(64, 32), dnn_activation="relu", dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
         dnn_dropout=0, output_activation="linear", seed=1024):
    """Instantiates the MIND Model architecture

    :param user_feature_columns: An iterable containing user's features used by the model.
    :param item_feature_columns: An iterable containing item's features used by the model.
    :param num_sampled: int, the number of classes to randomly sample per batch. (usually: randomly sample in recommend)
    :param k_max: int, the max size of user interest embedding
    :param p:float, the parameter for adjusting the attention distribution in LabelAwareAttention
    :param dynamic_k: bool, whether or not use dynamic interest number
    :param dnn_use_bn: bool. whether use BatchNormalization before activation or not in deep net
    :param user_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layers of user tower
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn: L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0, 1), the probability we will drop out a given DNN coordinate.
    :param seed: integer, to use as random seed.
    :param output_activation: Activation function to use in output layer
    :return A keras model instance

    """

    if len(item_feature_columns) > 1:
        # question3: 在论文不是同时只是品牌id、种类id与item_id 等？
        raise ValueError("Now MIND only support 1 item feature like item_id")

    # question4: 代码中为什么不统一写法
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_column.vocabulary_size
    item_embedding_dim = item_feature_column.embedding_dim

    history_feature_list = [item_feature_name]  # ['movie_id']

    # user tensor
    features = build_input_features(user_feature_columns)
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSpareFeat), user_feature_columns)) if user_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    seq_max_len = history_feature_columns[0].maxlen

    # user embedding keras tensor
    inputs_list = list(features.values())
    # create user/item embedding initialization part
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed, prefix="")

    # create item tensor
    item_features = build_input_features(item_feature_columns)
    # lookup for item emb list(*)
    query_emb_list = embedding_lookup(embedding_matrix_dict, item_features, item_feature_columns,
                                      history_feature_list,
                                      history_feature_list, to_list=True)
    # lookup for user's sequence item_emb list(*)
    keys_emb_list = embedding_lookup(embedding_matrix_dict, features, user_feature_columns,
                                     history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    # lookup for dnn input user emb list(*)
    dnn_input_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)

    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns, to_list=True)




