import pandas as pd
import torch

from tools import get_scoring_result, get_config_info, check_result_dir, drawLoss, save_score, save_model


def train(epoch, model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss_train = 0
    y_train = []
    y_prob_train = []
    for idx_train, (X, y) in enumerate(train_loader, 1):
        try:
            X = X.to(device)
        except Exception as e:
            # print(e)
            pass
        y = y.float().to(device)
        # print(X.shape, y.shape)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()

        y_train.extend(y.to('cpu').tolist())
        y_prob_train.extend(y_hat.to('cpu').tolist())
        # if idx_train % 20 == 0:
        #     print(f'epoch {epoch} [{idx_train}/{len(train_loader)}] batch_loss = {loss.item()}')
    y_pred = [1 if item >= 0.5 else 0 for item in y_prob_train]
    print(f'epoch {epoch} [{idx_train}/{len(train_loader)}] total_loss = {total_loss_train / idx_train}',
          end=" ")
    if scheduler:
        print(f'lr = {scheduler.get_lr()[0]},', end=" ")
    scoring = ['roc_auc', 'average_precision', 'accuracy', 'matthews_corrcoef', 'f1']
    process_msg, train_score_dict = get_scoring_result(scoring, y_train, y_pred, y_prob_train)
    print(process_msg)
    train_loss_epoch = total_loss_train / idx_train * 1.0
    return train_loss_epoch, train_score_dict


def test(epoch, model, test_loader, criterion, device):
    model.eval()
    total_loss_test = 0
    y_test = []
    y_prob = []
    with torch.no_grad():
        for idx_test, (X, y) in enumerate(test_loader, 1):
            try:
                X = X.to(device)
            except Exception as e:
                # print(e)
                pass
            y = y.float().to(device)
            # print(X.shape, y.shape)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            total_loss_test += loss.item()

            y_test.extend(y.to('cpu').tolist())
            y_prob.extend(y_hat.to('cpu').tolist())
            # if idx_test % 20 == 0:
            #     print(f'epoch {epoch} [{idx_test}/{len(train_loader)}] batch_loss = {loss.item()}')
        y_pred = [1 if item >= 0.5 else 0 for item in y_prob]
        print(f'epoch {epoch} [{idx_test}/{len(test_loader)}] total_loss = {total_loss_test / idx_test}',
              end=" ")
        scoring = ['roc_auc', 'average_precision', 'accuracy', 'matthews_corrcoef', 'f1']
        process_msg, test_score_dict = get_scoring_result(scoring, y_test, y_pred, y_prob)
        print(process_msg)
        test_loss_epoch = total_loss_test / idx_test * 1.0
        return test_loss_epoch, test_score_dict


def model_fit(config, model, train_loader, test_loader, optimizer, criterion, scheduler=None, threshold=0.86,
              cv_idx=""):
    if cv_idx != "":
        if scheduler:
            print(f">>>>>>>>>>『CV START』[{str(config.cv)}/{str(cv_idx)}-->lr:{scheduler.get_lr()}]>>>>>>>>>>>>")
        print(f">>>>>>>>>>『CV START』[{str(config.cv)}/{str(cv_idx)}]>>>>>>>>>>>>")
    get_config_info(config)
    train_loss = []
    test_loss = []
    test_score = []
    train_score = []
    lr_list = []
    check_result_dir(config, cv_idx=cv_idx)
    for epoch in range(1, config.epochs + 1):
        train_loss_epoch, train_score_dict = train(epoch, model, train_loader, optimizer, criterion, scheduler, config.device)
        if scheduler:
            lr_list.append(scheduler.get_lr()[0])
            scheduler.step()  # update lr
        test_loss_epoch, test_score_dict = test(epoch, model, test_loader, criterion, config.device)
        if test_score_dict["accuracy"] > threshold:
            save_model(config, model, epoch, cv_idx)
            threshold = test_score_dict["accuracy"]
        print("")

        train_score_dict.update({'loss': train_loss_epoch})
        train_score_dict.update({'lr': lr_list})
        test_score_dict.update({'loss': test_loss_epoch})
        train_score.append(train_score_dict)
        test_score.append(test_score_dict)

        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)

    train_score_df = pd.DataFrame.from_dict(train_score)
    test_score_df = pd.DataFrame.from_dict(test_score)

    drawLoss(config, train_loss, test_loss, cv_idx=cv_idx)

    print("『TRAIN__TEST OVER!!!』")
    if cv_idx != "":
        print(f">>>>>>>>>>『CV END』[{str(config.cv)}/{str(cv_idx)}]>>>>>>>>>>>>")
    print(" ")
    get_config_info(config)
    save_score(config, train_score_df, test_score_df, cv_idx=cv_idx)
