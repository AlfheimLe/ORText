import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

if __name__ == '__main__':
    data_name = 'Baby'
    num = 1
    auc_file = data_name + str(num) + '.pkl'
    with open(os.path.join('./result/POR', data_name, auc_file), 'rb') as f:
        train_porauc = pickle.load(f)
        test_porauc = pickle.load(f)
        portime = pickle.load(f)
        pormse = pickle.load(f)
        pormae = pickle.load(f)
    with open(os.path.join('./result/FastText', data_name, auc_file), 'rb') as f:
        train_mcauc = pickle.load(f)
        test_mcauc = pickle.load(f)
        mctime = pickle.load(f)
        mcmse = pickle.load(f)
        mcmae = pickle.load(f)
    with open(os.path.join('./result/CORAL', data_name, auc_file), 'rb') as f:
        train_coauc = pickle.load(f)
        test_coauc = pickle.load(f)
        cotime = pickle.load(f)
        comse = pickle.load(f)
        comae = pickle.load(f)
    with open(os.path.join('./result/OR', data_name, auc_file), 'rb') as f:
        train_or2auc = pickle.load(f)
        test_or2auc = pickle.load(f)
        or2time = pickle.load(f)
        or2mse = pickle.load(f)
        or2mae = pickle.load(f)
    # with open(os.path.join('./result/CNNOR', data_name, auc_file), 'rb') as f:
    #     train_cnnor2auc = pickle.load(f)
    #     test_cnnor2auc = pickle.load(f)
    #     cnnor2time = pickle.load(f)

    plt.figure()

    # plt.subplot(211)
    # plt.plot(list(range(len(test_mcauc))), test_mcauc, label='MCC')
    # plt.plot(list(range(len(test_coauc))), test_coauc, label='CORAL')
    # plt.plot(list(range(len(test_or2auc))), test_or2auc, label='OR')
    # # plt.plot(list(range(len(test_cnnor2auc))), test_cnnor2auc, label='CNNPOR')
    # plt.plot(list(range(len(test_porauc))), test_porauc, label='Ours')
    #
    # plt.ylabel('AUC')
    # plt.xlabel('Iteration x100 ')
    # plt.grid()
    # plt.legend()

    plt.subplot(311)
    plt.plot(list(range(len(test_mcauc))), test_mcauc, label='MCC')
    plt.plot(list(range(len(test_coauc))), test_coauc, label='CORAL')
    plt.plot(list(range(len(test_or2auc))), test_or2auc, label='OR')
    # plt.plot(cnnor2time, test_cnnor2auc, label='CNNPOR')
    plt.plot(list(range(len(test_porauc))), test_porauc, label='Ours')
    plt.ylabel('AUC')
    plt.xlabel('Iters X100')
    plt.grid()
    plt.legend()

    plt.subplot(312)
    plt.plot(list(range(len(mcmse))), mcmse, label='MCC')
    plt.plot(list(range(len(comse))), comse, label='CORAL')
    plt.plot(list(range(len(or2mse))), or2mse, label='OR')
    # plt.plot(cnnor2time, test_cnnor2auc, label='CNNPOR')
    plt.plot(list(range(len(pormse))), pormse, label='Ours')
    plt.ylabel('MSE')
    plt.xlabel('Iters X100')
    plt.grid()
    plt.legend()

    plt.subplot(313)
    plt.plot(list(range(len(mcmae))), mcmae, label='MCC')
    plt.plot(list(range(len(comae))), comae, label='CORAL')
    plt.plot(list(range(len(or2mae))), or2mae, label='OR')
    # plt.plot(cnnor2time, test_cnnor2auc, label='CNNPOR')
    plt.plot(list(range(len(pormae))), pormae, label='Ours')
    plt.ylabel('MAE')
    plt.xlabel('Iters X100')
    plt.grid()
    plt.legend()

    plt.show()
