import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import sys
sys.path.append("../DTINet")

from DTINet import DTINet

DATA_DIR = '../data'

def prepare_similarity_networks():
    drug_association_files = ["mat_drug_drug.txt"]#, "mat_drug_disease.txt", "mat_drug_se.txt"]
    prot_association_files = ["mat_protein_protein.txt"]#, "mat_protein_disease.txt"]
    drug_structure_sim = "Similarity_Matrix_Drugs.txt"
    prot_sequence_sim = "Similarity_Matrix_Proteins.txt"

    drug_sim_nets = []
    for f in drug_association_files:
        mat = np.loadtxt(DATA_DIR + "/" + f)
        mat = 1 - pdist(mat, 'jaccard')
        mat = squareform(mat)
        drug_sim_nets.append(mat)
    drug_sim_nets.append(np.loadtxt(DATA_DIR + "/" + drug_structure_sim))

    prot_sim_nets = []
    for f in prot_association_files:
        mat = np.loadtxt(DATA_DIR + "/" + f)
        mat = 1 - pdist(mat, 'jaccard')
        mat = squareform(mat)
        prot_sim_nets.append(mat)
    prot_sim_nets.append(np.loadtxt(DATA_DIR + "/" + prot_sequence_sim))

    return drug_sim_nets, prot_sim_nets


def split_interaction_file(ratio=1, seed=42):
    M = np.loadtxt(DATA_DIR + "/mat_drug_protein.txt")
    pos_idx = np.array(np.where(M == 1)).T
    neg_idx = np.array(np.where(M == 0)).T
    _rng = np.random.RandomState(seed)
    neg_sampled_idx = _rng.choice(neg_idx.shape[0], int(pos_idx.shape[0] * ratio), replace=False)
    neg_idx = neg_idx[neg_sampled_idx]

    train_pos_idx, test_pos_idx = train_test_split(pos_idx, test_size=0.2, random_state=seed)
    train_neg_idx, test_neg_idx = train_test_split(neg_idx, test_size=0.2, random_state=seed)

    train_idx = np.concatenate((train_pos_idx, train_neg_idx), axis=0)
    train_P = np.zeros(M.shape)
    train_P[train_idx[:, 0], train_idx[:, 1]] = 1

    test_idx = np.concatenate((test_pos_idx, test_neg_idx))
    test_y = np.concatenate((np.ones(test_pos_idx.shape[0]), np.zeros(test_neg_idx.shape[0])))

    return train_P, test_idx, test_y


def evaluate(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc


if __name__ == '__main__':
    print("Preparing interaction data...")
    train_P, test_idx, y_true = split_interaction_file()
    print("Computing similarity networks...")
    drug_sim_nets, prot_sim_nets = prepare_similarity_networks()

    dtinet = DTINet(
        drug_sim_nets=drug_sim_nets,
        prot_sim_nets=prot_sim_nets,
        label_mat=train_P,
        dim_drug=100,
        dim_prot=400,
        imc_dim=50,
        imc_lambda=1,
        imc_solver_type=10,
        imc_max_iter=10,
        imc_threads=4,
        imc_seed=42,
        rwr_rsp=0.5)

    dtinet.train()
    y_pred = dtinet.predict(test_idx)

    roc_auc, pr_auc = evaluate(y_true, y_pred)
    print "AUROC: %.4f, AUPRC: %.4f" % (roc_auc, pr_auc)