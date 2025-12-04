import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from regression_metrics import cal_regression_metrics

# def merge(file, num_tasks, merge_probs = True):
#     dict_feats = {}
#     dict_label = {}
#     dict_pos = {}
# 
#     for _ in range(num_tasks):
#         lines = open(file, 'r').readlines()[1:]
#         for line in lines:
#             line = line.strip()
#             name = line.split('[')[0]
#             label = np.fromstring(line.split(']')[1].split('[')[1], dtype=np.float32, sep=',')
#             chunk_nb = line.split(']')[2].split(' ')[1]
#             split_nb = line.split(']')[2].split(' ')[2]
#             data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float32, sep=',')
#             if merge_probs:
#                 data = softmax(data)
#             if not name in dict_feats:
#                 dict_feats[name] = []
#                 dict_label[name] = 0
#                 dict_pos[name] = []
#             if chunk_nb + split_nb in dict_pos[name]:
#                 continue
#             dict_feats[name].append(data)
#             dict_pos[name].append(chunk_nb + split_nb)
#             dict_label[name] = label
# 
#     input_lst = []
#     # more metrics and save preds
#     pred_dict = {'id': [], 'label': [], 'pred': []}
#     for i, item in enumerate(dict_feats):
#         input_lst.append([i, item, dict_feats[item], dict_label[item]])
#         pred = np.mean(dict_feats[item], axis=0)
#         if not merge_probs:
#             pred = softmax(pred)
#         label = dict_label[item]
#         pred_dict['pred'].append(pred)
#         pred_dict['label'].append(label)
#         pred_dict['id'].append(item.strip())
# 
#     return pred_dict

def merge(file, num_tasks):
    print(f"DEBUG: {num_tasks}")
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = np.fromstring(line.split(']')[1].split('[')[1], dtype=np.float64, sep=',')
            chunk_nb = line.split(']')[2].split(' ')[1]
            split_nb = line.split(']')[2].split(' ')[2]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float64, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    # more metrics and save preds
    pred_dict = {'id': [], 'label': [], 'pred': []}
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        pred = np.mean(dict_feats[item], axis=0)
        label = dict_label[item]
        pred_dict['pred'].append(pred)
        pred_dict['label'].append(label)
        pred_dict['id'].append(item.strip())

    # calculate overall metrics on the test set
    total_preds, total_labels = np.stack(pred_dict['pred']), np.stack(pred_dict['label'])
    metrics_dict = cal_regression_metrics(total_preds, total_labels, return_finegrained=True)

    return metrics_dict, pred_dict


def extract(epochs, loss):
    for split in range(1,6):
        in_dir = f"./saved/model/finetuning/blemore/audio_visual/voxceleb2_hicmae_pretrain_base/checkpoint-99/eval_split0{split}_lr_1e-3_epoch_{epochs}_size160_a256_sr4_server_loss_{loss}/"
        out_dir = f"./final_outputs/{loss}_{epochs}/split0{split}"
        print(in_dir)
        for i in range(epochs):
            in_file = os.path.join(in_dir, f"0_{i}.txt")
            if not os.path.exists(in_file):
                in_file = os.path.join(in_dir, f"0{i}.txt")

            _, pred_dict = merge(in_file, 1)
            df = pd.DataFrame(pred_dict)
            os.makedirs(out_dir, exist_ok=True)
            file = os.path.join(out_dir, f"ep_{i}_prob_merge.csv")
            df.to_csv(file)

if "__main__" == __name__:
    # epoch_count = [20, 50, 100]
    epoch_count = [80]
    loss_names = ["ce"]
    for epochs in epoch_count:
        for loss in loss_names:
            extract(epochs, loss)
