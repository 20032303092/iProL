import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader

from feature_extraction import Kmer

_current_path = os.path.dirname(__file__)


def get_excel(version='', sheet_name='dataset'):
    """
    version candidate: 10.10, 9.3
    sheet_name(dataset) candidate: dataset+u-cdhit
    :param version:
    :param sheet_name:
    :return: a new sheet include 'Pro_seq', 'Sigma' and 'Con_level'
    """
    data_root = r'../data/'
    # version = '9.3'
    xlsx_file = data_root + '/RegulonDB(Version' + version + ')/dataset' + version + '.xlsx'
    xlsx_file = os.path.join(_current_path, xlsx_file)
    # sheet_name = 'dataset9.3'
    sheet = pd.read_excel(xlsx_file, sheet_name=sheet_name, engine='openpyxl')
    columns = sheet.columns
    # print(columns)
    sheet = sheet[['Pro_seq', 'Sigma', 'Con_level']]
    return sheet


def get_complete_data(sheet):
    """
    add column 'Pro'
    add column 'Activity'
    :param sheet:
    :return: a new sheet include 'Pro_seq', 'Sigma', 'Con_level' and 'Pro'
    """
    sheet.loc[sheet['Sigma'] == 'non_pro', 'Pro'] = '0'
    sheet.loc[sheet['Sigma'] != 'non_pro', 'Pro'] = '1'
    sheet['Con_level'] = sheet['Con_level'].fillna("non_level")
    return sheet


def get_datasets(complete_data):
    """
    get three datasets: train_identify, train_type, train_level
    :param complete_data:
    :return:
    """

    """
    Step1 get dataset to predict Pro/non_pro
    """
    dataset_identify = complete_data[['Pro_seq', 'Pro']].copy()

    # del non_pro to get promoters
    complete_data = __del_rows(complete_data, "Sigma", ["non_pro"])

    """
    Step2 get dataset to predict pro level
    """
    # del confirmed in column Con_level
    complete_data = __del_rows(complete_data, "Con_level", "Confirmed")
    dataset_level = complete_data[['Pro_seq', 'Con_level']].copy()

    """
    Step3 get dataset to predict pro type
    """
    # del unknown in column Sigma
    complete_data = __del_rows(complete_data, "Sigma", ["unknown"])
    dataset_type = complete_data[['Pro_seq', 'Sigma']].copy()

    """
    step4 处理成可输入格式
    """
    dataset_identify["Pro_seq"] = dataset_identify["Pro_seq"].str.upper()
    dataset_identify["Pro"] = dataset_identify["Pro"].astype("int")
    dataset_level["Pro_seq"] = dataset_level["Pro_seq"].str.upper()
    dataset_level["Con_level"] = dataset_level["Con_level"].replace(["Strong", "Weak"], [1, 0])
    sigma_list = ['Sigma70', 'Sigma24', 'Sigma32', 'Sigma38', 'Sigma28', 'Sigma54']
    label_list = [0, 1, 2, 3, 4, 5]
    dataset_type["Pro_seq"] = dataset_type["Pro_seq"].str.upper()
    dataset_type["Sigma"] = dataset_type["Sigma"].replace(sigma_list, label_list)

    return dataset_identify, dataset_type, dataset_level


def get_cv_dataloader_list(config, dataset, pred):
    """
    :param dataset:
    :param pred:
    :param cv:
    :param random_state:
    :return:
    """
    # pred: 'Pro'|'Con_level'|'Sigma'
    skf = StratifiedKFold(n_splits=config.cv, shuffle=True, random_state=config.random_state)
    # dataset_list format: [[train,test],[cv2],[cv3],...,[cvn]]
    dataset_list = __get_cv_dataset_list(dataset, skf, pred)

    # loader_list format: [[train_loader,test_loader],[cv2],[cv3],...,[cvn]]
    loader_list = []
    for single_cv_dataset in dataset_list:
        single_cv_loader = _feature_exaction(config=config, dataset=single_cv_dataset, pred=pred)
        loader_list.append(single_cv_loader)
    return dataset_list, loader_list


def _feature_exaction(config, dataset, pred):
    dataset_train, dataset_test = dataset
    kmer = Kmer(k=[config.kmer], stride=1, return_type='seq')
    print(f"PBC[{config.pc_name}] --- kmer = {config.kmer}")

    train_kmer_seq = kmer.run_fea(dataset_train["Pro_seq"].tolist())
    test_kmer_seq = kmer.run_fea(dataset_test["Pro_seq"].tolist())

    train_dataset = RegulonDataset(train_kmer_seq, dataset_train[pred])
    test_dataset = RegulonDataset(test_kmer_seq, dataset_test[pred])
    # print(train_dataset[1], test_dataset[1])
    print("train len:", len(train_dataset), "test len:", len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=8)
    return [train_loader, test_loader]


def feature_exaction(config, dataset, pred):
    return _feature_exaction(config, dataset, pred)


def __del_rows(dataset: pd, column, value):
    if isinstance(value, str):
        value = [value]
    # print('del:', column, value)
    dataset_new = dataset[~dataset[column].isin(value)]
    return dataset_new


def get_first_level_train_test(sheet, random_state=None, independent_test_size=0.2):
    """
    to make same train data and test data in three tasks,
    and then we will do next step split by specific split type
    :param sheet: the complete data (N, 4)
    :param random_state:
    :param independent_test_size:
    :return: train data (N1, 4), and independent test data (N2, 4)
    """

    """
    Step1 split base_data to train and independent test
    """
    train, independent_test, _, _ = train_test_split(sheet, sheet[['Pro']],
                                                     stratify=sheet[['Pro']],
                                                     random_state=random_state,
                                                     shuffle=True, test_size=independent_test_size)
    return [train, independent_test]


def _get_dataset_by_train_test(first_level_train_test):
    """
    Step1 split base_data to train and independent test
    """
    # train, independent_test = get_first_level_train_test(random_state=random_state,
    #                                                      independent_test_size=independent_test_size)
    if len(first_level_train_test) != 2:
        raise ValueError("first level dataset must include train and independent_test")
    train, independent_test = first_level_train_test[0], first_level_train_test[1]
    """
    Step2 get dataset from `train` and `independent_test` to predict Pro/non_pro
    """
    train_identify = train[['Pro_seq', 'Pro']]
    independent_test_identify = independent_test[['Pro_seq', 'Pro']]

    # del non_pro
    train = __del_rows(train, "Sigma", ["non_pro"])
    independent_test = __del_rows(independent_test, "Sigma", ["non_pro"])

    """
    Step4 get dataset from `train` and `independent_test` to predict pro level
    """
    # del confirmed in column Con_level
    train = __del_rows(train, "Con_level", "Confirmed")
    independent_test = __del_rows(independent_test, "Con_level", "Confirmed")
    train_level = train[['Pro_seq', 'Con_level']]
    independent_test_level = independent_test[['Pro_seq', 'Con_level']]

    """
    Step3 get dataset from `train` and `independent_test` to predict pro type
    """
    # del unknown and non_pro in column Sigma
    train = __del_rows(train, "Sigma", ["unknown"])
    independent_test = __del_rows(independent_test, "Sigma", ["unknown"])
    train_type = train[['Pro_seq', 'Sigma']]
    independent_test_type = independent_test[['Pro_seq', 'Sigma']]

    return [train_identify, independent_test_identify], \
        [train_type, independent_test_type], \
        [train_level, independent_test_level]


def _get_dataset_by_train_val_test(first_level_train_test, random_state=None, val_test_size=0.5):
    # train, independent_test = get_first_level_train_test(random_state=random_state,
    #                                                      independent_test_size=independent_test_size)
    if len(first_level_train_test) != 2:
        raise ValueError("first level dataset must include train and independent_test")
    train, independent_test = first_level_train_test[0], first_level_train_test[1]

    """
    Step1 train set is not changed, split independent_test into val and test_new
    """
    # drop single sample in a composition class
    Confirmed_test_X = independent_test[independent_test['Con_level'] == 'Confirmed']
    if Confirmed_test_X.size != 0:
        index1 = Confirmed_test_X[Confirmed_test_X.Sigma == 'unknown'].index.tolist()[0]
        independent_test.drop(index=[index1], inplace=True)
    # split
    val, independent_test, _, _ = train_test_split(independent_test,
                                                   independent_test[['Sigma', 'Con_level', 'Pro']],
                                                   stratify=independent_test[['Sigma', 'Con_level', 'Pro']],
                                                   random_state=random_state,
                                                   shuffle=True, test_size=val_test_size)

    """
    Step2 get dataset from `train` and `independent_test` to predict Pro/non_pro
    """
    train_identify = train[['Pro_seq', 'Pro']]
    val_identify = val[['Pro_seq', 'Pro']]
    independent_test_identify = independent_test[['Pro_seq', 'Pro']]

    """
    Step3 del non_pro
    """
    train = __del_rows(train, "Sigma", ["non_pro"])
    val = __del_rows(val, "Sigma", ["non_pro"])
    independent_test = __del_rows(independent_test, "Sigma", ["non_pro"])

    """
    Step4 get dataset from `train` and `independent_test` to predict pro level
    """
    # del confirmed in column Con_level
    train = __del_rows(train, "Con_level", "Confirmed")
    val = __del_rows(val, "Con_level", "Confirmed")
    independent_test = __del_rows(independent_test, "Con_level", "Confirmed")
    train_level = train[['Pro_seq', 'Con_level']]
    val_level = val[['Pro_seq', 'Con_level']]
    independent_test_level = independent_test[['Pro_seq', 'Con_level']]

    """
    Step5 get dataset from `train` and `independent_test` to predict pro type
    """
    train = __del_rows(train, "Sigma", ["unknown"])
    val = __del_rows(val, "Sigma", ["unknown"])
    independent_test = __del_rows(independent_test, "Sigma", ["unknown"])
    train_type = train[['Pro_seq', 'Sigma']]
    val_type = val[['Pro_seq', 'Sigma']]
    independent_test_type = independent_test[['Pro_seq', 'Sigma']]

    return [train_identify, val_identify, independent_test_identify], \
        [train_type, val_type, independent_test_type], \
        [train_level, val_level, independent_test_level]


def __get_cv_dataset_list(train, skf, pred_column):
    dataset_list = []
    for train_index, test_index in skf.split(train, train[pred_column]):
        # print("TRAIN:", train_index, "TEST:", test_index)
        t1_train = train.iloc[train_index]
        t1_test = train.iloc[test_index]

        # check split
        train_test_index = t1_train.index.to_list()
        train_test_index.extend(t1_test.index.to_list())
        # print("check[KFold split]:", train.shape[0], len(set(train_test_index)))
        if train.shape[0] != len(set(train_test_index)):
            raise ValueError("train.shape[0]  != len(set(train_test_index))")

        # format
        dataset_list.append([t1_train[['Pro_seq', pred_column]], t1_test[['Pro_seq', pred_column]]])
    # dataset.append(independent_test)
    return dataset_list


def _get_dataset_by_stratifyKFold(first_level_train_test, random_state=None, cv=10):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    """
    Step1 split base_data to train and independent test
    """
    # train, independent_test = get_first_level_train_test(random_state=random_state,
    #                                                      independent_test_size=independent_test_size)
    if len(first_level_train_test) != 2:
        raise ValueError("first level dataset must include train and independent_test")
    train, independent_test = first_level_train_test[0], first_level_train_test[1]

    """
    Step2 get 5-fold datasets from train set to predict pro/non-pro, independent_test set is not changed.
    dataset_identify format: [[train,test],[cv2],[cv3],...,[cvn],independent_test]
    """
    dataset_identify = __get_cv_dataset_list(train, skf, "Pro")
    dataset_identify.append(independent_test[["Pro_seq", "Pro"]])

    # get train_X_del_nonPro_unknown and independent_test_del_nonPro_unknown
    train = __del_rows(train, "Sigma", ["non_pro"])  # train_del_nonPro_unknown
    independent_test = __del_rows(independent_test, "Sigma",
                                  ["non_pro"])  # independent_test_del_nonPro_unknown
    # print(len(train), len(independent_test))
    """
    Step3 get 5-fold datasets from train set to predict pro level, independent_test set is not changed.
    dataset_level format: [[train,test],[cv2],[cv3],...,[cvn],independent_test]
    """
    # get train_del_nonPro_unknown_confirmed and independent_test_del_nonPro_unknown_confirmed
    train = __del_rows(train, "Con_level", "Confirmed")  # train_del_nonPro_unknown_confirmed
    independent_test = __del_rows(independent_test, "Con_level",
                                  "Confirmed")  # independent_test_del_nonPro_unknown_confirmed
    dataset_level = __get_cv_dataset_list(train, skf, "Con_level")
    dataset_level.append(independent_test[["Pro_seq", "Con_level"]])

    """
    Step2 get 5-fold datasets from train set to predict pro type, independent_test set is not changed.
    dataset_type format: [[train,test],[cv2],[cv3],...,[cvn],independent_test]
    """
    train = __del_rows(train, "Sigma", ["unknown"])  # train_del_unknown
    independent_test = __del_rows(independent_test, "Sigma",
                                  ["unknown"])  # independent_test_del_unknown
    dataset_type = __get_cv_dataset_list(train, skf, "Sigma")

    dataset_type.append(independent_test[["Pro_seq", "Sigma"]])

    return dataset_identify, dataset_type, dataset_level


def get_split_dataset(complete_data, split_type='train_test', random_state=None,
                      independent_test_size=0.2, val_test_size=0.5,
                      cv=10):
    """
    obtain datasets of 3 tasks through data (N,4) and split type
    :param complete_data: the whole data with shape (N,4)
    :param split_type: 'train_test','train_val_test' or 'KFold'
    :param random_state:
    :param independent_test_size: first level split: keep same train data and independent test data
    :param val_test_size: second level split: get val and test, or KFold from independent_test of first level
    :param cv: KFold n_splits
    :return: 3 datasets
    """
    # first level split
    first_level_train_test = get_first_level_train_test(complete_data, random_state, independent_test_size)
    # second level split
    if split_type == "train_test":
        dataset_identify, dataset_type, dataset_level = _get_dataset_by_train_test(first_level_train_test)
    elif split_type == "train_val_test":
        # train set is not changed, split independent_test into val and test_new
        dataset_identify, dataset_type, dataset_level = _get_dataset_by_train_val_test(first_level_train_test,
                                                                                       random_state,
                                                                                       val_test_size)
    elif split_type == "KFold":
        dataset_identify, dataset_type, dataset_level = _get_dataset_by_stratifyKFold(first_level_train_test,
                                                                                      random_state,
                                                                                      cv)
    else:
        raise ValueError(f"split type error, {split_type} is not in ['train_test','train_val_test','KFold']!")

    return dataset_identify, dataset_type, dataset_level


class RegulonDataset(Dataset):
    def __init__(self, X, y):
        super(RegulonDataset, self).__init__()
        # self.X = np.array(X)
        # self.y = np.array(y)
        try:
            self.X = np.array(X, np.float32)
            self.y = np.array(y, np.float32)
        except ValueError:
            """ kmer seq to bert-like model for embedding"""
            self.X = np.array(X)
            self.y = np.array(y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    # usage:
    sheet = get_excel(version='', sheet_name='dataset')
    complete_data = get_complete_data(sheet)
    # dataset_identify, dataset_type, dataset_level = get_split_dataset(complete_data,
    #                                                                   split_type="train_test",
    #                                                                   independent_test_size=0.2)
    dataset_identify, dataset_type, dataset_level = get_datasets(complete_data)
    import config_longformer_3070 as config

    identify_dataset_list, identify_loader_list = get_cv_dataloader_list(config, dataset_identify, "Pro")
    type_dataset_list, type_loader_list = get_cv_dataloader_list(config, dataset_type, "Sigma")
    level_dataset_list, level_loader_list = get_cv_dataloader_list(config, dataset_level, "Con_level")

    # # test RegulonDataset
    # import pandas as pd
    #
    # d = [[1, 1], [2, 0], [3, '1'], [4, 0], [5, 1], [6, 1], [7, 0], [8, 1], [9, 0], [10, 0]]
    # X = pd.DataFrame(data=d, index=[3, 5, 7, 16, 47, 4, 67, 35, 26, 23], columns=['x', 'y'])
    # y = pd.DataFrame(data=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    # data = RegulonDataset(X, y)
    # print(data[1])
