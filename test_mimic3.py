import torch
import torch.nn.functional as F
from eval.metrics import multi_label_metric, ddi_rate_score
import numpy as np
from tqdm import tqdm
import argparse
import os
from model.mymodel import Model
import time
import dill


def eval_one_epoch(model, data_eval, n_drug, device):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    for step, input_seq in enumerate(tqdm(data_eval)):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            syms = torch.tensor(adm[0]).to(device)
            scores = model.evaluate(syms, device=device)

            y_gt_tmp = np.zeros(n_drug)
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    ddi_rate = ddi_rate_score(smm_record, path='datasets/MIMIC3/ddi_A_final.pkl')
    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), 1.0 * med_cnt / visit_cnt, ddi_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test')
    parser.add_argument('--path', required=True, help='the path of model')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--device', default=0, type=int, help='the cuda id')
    parser.add_argument('--cpu', action='store_true', help='use cpu for infer')
    args = parser.parse_args()
    args.batch_size = 50
    args.dataset = 'MIMIC3'
    with open(os.path.join('datasets', args.dataset, 'data_test_patient.pkl'), 'rb') as Fin:
        data_test = dill.load(Fin)
    pklSet = PKLSet(args.batch_size, args.dataset)
    device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.device}')
    model = Model(pklSet.n_sym, pklSet.n_drug, torch.FloatTensor(pklSet.ddi_adj).to(device), pklSet.sym_sets,
                  torch.tensor(pklSet.drug_multihots).to(device), args.embedding_dim).to(device)
    np.random.seed(0)
    model.load_state_dict(torch.load(args.path, map_location=device))
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    for _ in range(10):
        test_sample = np.random.choice(data_test, sample_size, replace=True)
        ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = \
            eval_one_epoch(model, test_sample, pklSet.n_drug)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        print('-' * 89)
        print(
            '| best ja {:5.4f} | prauc {:5.4f} | avg_p {:5.4f} | avg_recall {:5.4f} | '
            'avg_f1 {:5.4f} | avg_med {:5.4f} | ddi_rate {:5.4f}'.format(ja, prauc, avg_p,
                                                                         avg_r,
                                                                         avg_f1, avg_med, ddi_rate))
        print('-' * 89)
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
