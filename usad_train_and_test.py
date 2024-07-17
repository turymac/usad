import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import os
import argparse
import torch.utils.data as data_utils

from sklearn import preprocessing
from utils import *
from usad import *

device = get_default_device()


def get_df(filepaths_csv, drop_time = False, delimiter=";"):
    # Load dataframes
    print("Loading data.")
    # Make dataframes
    
    dfs = [pd.read_csv(filepath_csv, sep=delimiter) for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    # Sort columns by name !!!
    df = df.sort_index(axis=1)

    # Gestisci timestamp
    if drop_time:
        df.drop(["time"], axis=1, inplace=True)
    else:
      df['time'] = pd.to_datetime(df['time'])
    
    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)

    # Drop NaN
    df = df.dropna(axis=0)

    print("Loading data done.\n")

    return df


# Funzione per verificare se c'è almeno un collisions_init tra start e end
def label_window(row, collisions_init):
    return int(any((collisions_init >= row['start']) & (collisions_init <= row['end'])))


def get_args():
    parser = argparse.ArgumentParser("USAD model fot TSAD")

    parser.add_argument("--normal_path",
                        default="/content/drive/MyDrive/Kuka_v1/normal",
                        type=str,
                        help="Path for training data")

    parser.add_argument("--anomaly_path",
                        default="/content/drive/MyDrive/Kuka_v1/collisions",
                        type=str,
                        help="Path for test data")     

    parser.add_argument('--period',
                        type=str,
                        default="0.1",
                        help='Sampling rate') 
                        
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5,
                        help='alpha') 

    parser.add_argument('--beta',
                        type=float,
                        default=0.5,
                        help='beta')                                     
    
    parser.add_argument("--window_len",
                        default=12,
                        type=int,
                        help="length of a sliding window")

    parser.add_argument("--batch",
                        default=128,
                        type=int,
                        help="batch size")

    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs")

    parser.add_argument("--hidden_size",
                        default=50,
                        type=int,
                        help="dim of encoder hidden state")                        

    parser.add_argument("--verbose",
                        default=False,
                        type=bool,
                        help="enable/disable debug prints")                        

    args = parser.parse_args()
    
    accepted_periods = ["0.01", "0.1", "0.005", "1.0"]
    assert args.period in accepted_periods, f"Chosen period is not correct"

    if args.alpha + args.beta != 1:
      ags.beta = 1 - args.alpha
      print(f"WARNING: set beta to {beta}")
    
    return args


def main():
    args = get_args()

    # if args.script_mode.startswith('train'):
    #     # launch trainer
    #     print("training...")
    #     assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
    #     assert args.exp_name is not None and len(args.exp_name)
    #     args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
    #     args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
    #     args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
    #     args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
    #     train(args, config)
    # else:
    #     # eval Modelnet -> SONN
    #     assert args.ckpt_path is not None and len(args.ckpt_path)
    #     print("out-of-distribution eval - Modelnet -> SONN ..")
    #     eval_ood_md2sonn(args, config)
    train_and_eval(args)

def train_and_eval(args) :

    # ROOTDIR_DATASET_NORMAL = "/content/drive/MyDrive/Kuka_v1/normal"
    # period = 0.1

    filepath_csv = [os.path.join(args.normal_path, f"rec{r}_20220811_rbtc_{args.period}s.csv") for r in [0, 2, 3, 4]]
    
    #Read data
    normal = get_df(filepath_csv, drop_time = True)

    if args.verbose:
      print(f"Training data shape: {normal.shape}")

    min_max_scaler = preprocessing.MinMaxScaler()

    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)

    if args.verbose:
      print(f"Training data normalized: {normal.head(2)}")

    # ROOTDIR_DATASET_ANOMALY = "/content/drive/MyDrive/Kuka_v1/collisions"

    filepath_csv = [os.path.join(args.anomaly_path, f"rec{r}_collision_20220811_rbtc_{args.period}s.csv") for r in [1, 5]]

    #Read data
    df = get_df(filepath_csv, drop_time = False)

    attack = df.drop(columns=['time'], axis=1)
    
    if args.verbose:
      print(f"Test data shape: {attack.shape}")

    x = attack.values
    x_scaled = min_max_scaler.transform(x)
    attack = pd.DataFrame(x_scaled)

    if args.verbose:
      print(f"Training data normalized: {attack.head(2)}")

    # window_size=6

    windows_normal=normal.values[np.arange(args.window_len)[None, :] + np.arange(normal.shape[0]-args.window_len +1)[:, None]]
    
    if args.verbose:
      print(f"Training windows shape: {windows_normal.shape}")

    windows_attack = np.array([attack[i:i + args.window_len] for i in range(len(attack) - args.window_len + 1)])
    windows_attack.shape

    if args.verbose:
      print(f"Test windows shape: {windows_attack.shape}")

    collisions = pd.read_excel(os.path.join(args.anomaly_path, "20220811_collisions_timestamp.xlsx"))
    collisions_init = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta([2] * len(collisions[collisions['Inizio/fine'] == "i"].Timestamp), 'h')

    # Estrai i timestamp iniziali e finali per le finestre
    timestamps_start = df['time'].values[:len(attack) - args.window_len + 1]
    timestamps_end = df['time'].values[args.window_len - 1:]

    # Crea un DataFrame con i timestamp iniziali e finali
    timestamps_info = pd.DataFrame({
        'start': timestamps_start,
        'end': timestamps_end
    })

    # Applicare la funzione per creare la colonna 'label'
    timestamps_info['label'] = timestamps_info.apply(label_window, axis=1, collisions_init=collisions_init)

    if args.verbose:
      print(f"Labels windows shape: {timestamps_info.shape}")
      # Stampa i risultati per verifica
      print(timestamps_info[timestamps_info['label'] == 0].head(5))
      print(timestamps_info[timestamps_info['label'] == 1].head(5))

    w_size=windows_normal.shape[1]*windows_normal.shape[2]
    z_size=windows_normal.shape[1]*args.hidden_size

    windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ) , batch_size=args.batch, shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ) , batch_size=args.batch, shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
    ) , batch_size=args.batch, shuffle=False, num_workers=0)

    model = UsadModel(w_size, z_size)
    model = to_device(model,device)

    history = training(args.epochs,model,train_loader,val_loader)

    print("---Loss History plot---")
    plot_history(history)

    torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
                }, "model.pth")

    checkpoint = torch.load("model.pth")

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    results=testing(model,test_loader,alpha=args.alpha,beta=args.beta)

    y_test = timestamps_info['label'].values

    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                  results[-1].flatten().detach().cpu().numpy()])

    if args.verbose:                                
      print(f"Y_test shape: {y_test.shape}")
      print(f"Y_pred shape: {y_pred.shape}")  


    threshold=ROC(y_test,y_pred)

if __name__ == '__main__':
    main()