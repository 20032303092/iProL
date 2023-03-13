import math
import os
import time
from inspect import signature

import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def _get_score_dict(scoring):
    score_dict = {}
    if isinstance(scoring, str):
        score_dict.update({scoring + '_score': scoring})
    else:
        for item in scoring:
            score_dict.update({item + '_score': item})
    # score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
    # print(score_dict)
    return score_dict


def get_scoring_result(scoring, y, y_pred, y_prob, y_score=None, is_base_score=True):
    process_msg = ""
    if y_score is None:
        y_score = y_prob
    module_name = __import__("sklearn.metrics", fromlist='*')
    # print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
    score_dict = _get_score_dict(scoring)
    # print(score_dict)
    # start get_scoring_result
    score_result_dict = {}
    if is_base_score:
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        """
        Sensitivity  Sn = TP / (TP + FN)
        Specificity  Sp = TN / (TN + FP)
        """
        Sn = TP / (TP + FN) * 1.0
        Sp = TN / (TN + FP) * 1.0
        score_result_dict.update({"Total": len(y), "TP": TP, "TN": TN, "FP": FP, "FN": FN, "precision": precision,
                                  "recall": recall, "Sn": Sn, "Sp": Sp})
        process_msg += "total=%s, TP=%s, TN=%s, FP=%s, FN=%s; precision=%.3f, recall=%.3f, Sn=%.3f, Sp=%.3f\n" \
                       % (len(y), TP, TN, FP, FN, precision, recall, Sn, Sp)
    for k, v in score_dict.items():
        # print("===", k)
        try:
            score_func = getattr(module_name, k)
        except AttributeError:
            score_func = getattr(module_name, k.split('_score')[0])

        sig = signature(score_func)
        # print(sig)
        y_flag = str(list(sig.parameters.keys())[1])
        # print(y_flag)
        if y_flag == 'y_pred':
            y_flag = y_pred
        elif y_flag == 'y_prob':
            y_flag = y_prob
        elif y_flag == 'y_score':
            y_flag = y_score
        else:
            raise ValueError("having new metrics that its 2nd param is not y_pred y_prob or y_score in sklearn !!!")
        if y_flag is None:
            raise ValueError(k, "%s is None !!!" % (y_flag))
        score_result = score_func(y, y_flag)
        # accuracy: (test=0.926)
        # print("%s: (test=%s)" % (v, score_result), end=" ")
        process_msg += "%s: (test=%.3f) " % (v, score_result)
        # print("%s: (test=%.3f) ===" % (v, score_result))
        score_result_dict.update({v: score_result})
    # print("score_result_dict:", score_result_dict)
    return process_msg, score_result_dict


def time_since(start):
    s = time.time() - start
    # s = 62 - start
    if s < 60:
        return '%.2fs' % s
    elif 60 < s and s < 3600:
        s = s / 60
        return '%.2fmin' % s
    else:
        m = math.floor(s / 60)
        s -= m * 60
        h = math.floor(m / 60)
        m -= h * 60
        return '%dh %dm %ds' % (h, m, s)


def get_data_np_dict(train_X, train_y, test_X, test_y):
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def drawLoss(config, train_loss, test_loss, train_acc=None, test_acc=None, cv_idx=""):
    # polt
    x = range(1, len(train_loss) + 1)
    plt.plot(x, train_loss, 'g-', label="train_loss")
    plt.plot(x, test_loss, 'r-', label="test_loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(config.ex_result_dir, str(cv_idx), config.loss_pic_name), dpi=300)
    plt.close()
    # plt.show()

    # plt.plot(x, loss_list, 'c-', label="train_loss")
    # # plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    # plt.ylabel("aupr")
    # plt.xlabel("epoch")
    # plt.legend()
    # plt.savefig("../model/%s_%s_%s_train_loss.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.close()
    # plt.show()

    # plt.plot(x, test_auc_list, 'g-', label="test_auc")
    # plt.plot(x, train_auc_list, 'r:', label="train_auc")
    # plt.plot(x, test_aupr_list, 'c-', label="test_aupr")
    # plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    # plt.ylabel("auc & aupr")
    # plt.xlabel("epoch")
    # plt.legend()
    # plt.savefig("../model/%s_%s_%s_epoch_auc&aupr.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.close()
    # plt.show()


def save_model(config, model, epoch, cv_idx=""):
    dir = os.path.join(config.ex_result_dir, str(cv_idx), "model")
    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(model.state_dict(), os.path.join(dir, f'model_{epoch}.pkl'))
    print("『saved model』:", os.path.join(dir, f'model_{epoch}.pkl'))
    # new_model = Model()  # 调用模型Model
    # new_model.load_state_dict(torch.load("./data/model_parameter.pkl"))  # 加载模型参数
    # new_model.forward(input)  # 进行使用


def save_score(config, train_score_df, test_score_df, cv_idx=""):
    dir = os.path.join(config.ex_result_dir, str(cv_idx))
    train_score_df.to_csv(os.path.join(dir, config.train_score_name), index=True)
    test_score_df.to_csv(os.path.join(dir, config.test_score_name), index=True)

    print("『Score Saving』Score Saved!!!")


def check_result_dir(config, cv_idx=""):
    cv_ex_result_dir = os.path.join(config.ex_result_dir, str(cv_idx))
    # check path
    if os.path.exists(cv_ex_result_dir):
        if os.path.exists(os.path.join(cv_ex_result_dir, config.loss_pic_name)):
            raise FileExistsError(f"Ex result dir {config.ex_result_dir} has been used, please check!")
    else:
        os.makedirs(cv_ex_result_dir)
        print(f"Ex result dir {cv_ex_result_dir} has been created!")


def get_config_info(config):
    print(f"『Computer device』nvidia:{config.pc_name}")
    print(f"『dataset』kmer: {config.kmer}, cv: {config.cv}")
    print(
        f"『train config』epochs: {config.epochs}, lr: {config.lr}, batch_size: {config.batch_size}"
        f", device: {config.device}")
    if config.pre_model:
        print(
            f"『pre-model config [{config.pre_model}]』add_special_tokens: {config.add_special_tokens}"
            f", not_fine_tuning: {config.not_fine_tuning}, model_name: {config.pre_model_file}")
    if config.dna2vec:
        print(f"『dna2vec config [{config.dna2vec}]』ebd_kmer: {str(config.ebd_kmer)}, end_name: {config.ebd_file}")
    if config.lstm:
        print(f"『lstm config [{config.lstm}]』"
              f"bidirectional: {str(config.bidirectional)}, num_layers: {str(config.num_layers)}")
    print(f"『ex result config』ex_result_dir: {config.ex_result_dir}")


def configTest(config):
    print(config.__file__)
    get_config_info(config)
    var = config.loss_pic_name
    print(var)
    print(config.ex_result_dir)


if __name__ == '__main__':
    print(sklearn.metrics.SCORERS.keys())
    """
    dict_keys(['explained_variance', 'r2', 'max_error', 
    'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
    'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_error', 
    'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 
    'accuracy', 'top_k_accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 
    'balanced_accuracy', 'average_precision', 'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'rand_score', 
    'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score',
     'normalized_mutual_info_score', 'fowlkes_mallows_score', 
     'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 
    'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 
    'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
     'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted']
    """
    scoring = ['roc_auc', 'recall', 'precision', 'matthews_corrcoef']
    y = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 1, 0]
    y_prob = [0.7, 0.5, 0.6, 0.6, 0.3, 0.2, 0.7, 0.9]
    process_msg, score_result_dict = get_scoring_result(scoring, y, y_pred, y_prob)
    print(process_msg)
