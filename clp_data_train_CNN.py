import pandas as pd

from dataset import *
from dataloader import *
from trainer import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from config import *
from utils import *
from model import *
import argparse
import torch

# set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`
# to enable the fallback to CPU tensors when CUDA tensors are out of memory.
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def train(args, export_root=None, resume=True):
    print(args.cutoff)
    data = pd.read_csv('data_use_144.csv')
    m, T = data.shape
    print(m)
    split = int(3*m/4)
    # label = '冷氣-2L2'
    label = args.appliance_names
    train_set = Mydataset(args, data[:split], label)
    # print(train_set.x.shape)
    # train_set.remove_null()
    # print(train_set.x.shape)
    test_set = Mydataset(args, data[split:], label)
    # test_set.remove_null()
    x_max = train_set.get_max()
    # print(x_max)
    train_set.standard()
    test_set.standard(x_max)

    x_mean_after, x_std_after = train_set.get_mean_std()
    stats = (x_mean_after, x_std_after)
    # print(x_mean_after)
    # print(x_std_after)
    model = LSTM(args)

    if export_root == None:
        # folder_name = '-'.join(args.appliance_names)
        folder_name = 'colling_2L1'
        export_root = 'experiments/' + args.dataset_code + '/' + folder_name

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    trainer = Trainer_CNN(args, model, train_loader,
                      val_loader, stats, export_root)
    if args.num_epochs > 0:
        # if resume:
            # try:
            #     model.load_state_dict(torch.load(os.path.join(
            #         export_root, 'best_acc_model.pth'), map_location='cpu'))
            #     print('Successfully loaded previous model, continue training...')
            # except FileNotFoundError:
            #     print('Failed to load old model, continue training new model...')
        trainer.train()


    rel_err, abs_err, acc, prec, recall, f1 = trainer.test(val_loader)
    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('Mean Relative Error:', rel_err)
    print('Mean Absolute Error:', abs_err)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)


torch.set_default_tensor_type(torch.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--dataset_code', type=str,
                    default='clp')
parser.add_argument('--validation_size', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--house_indicies', type=list, default=[1, 2, 3, 4, 5])
parser.add_argument('--appliance_names', type=list,
                    default=['microwave', 'dishwasher'])
parser.add_argument('--sampling', type=str, default='6s')
parser.add_argument('--cutoff', type=dict, default=None)
parser.add_argument('--threshold', type=dict, default=None)
parser.add_argument('--min_on', type=dict, default=None)
parser.add_argument('--min_off', type=dict, default=None)
parser.add_argument('--window_size', type=int, default=144)
parser.add_argument('--window_stride', type=int, default=6)
parser.add_argument('--normalize', type=str, default='mean',
                    choices=['mean', 'minmax'])
parser.add_argument('--denom', type=int, default=2000)
parser.add_argument('--model_size', type=str, default='gru',
                    choices=['gru', 'lstm', 'dae'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.25)
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str,
                    default='adam', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--c0', type=dict, default=None)
parser.add_argument('--is_od', type=bool, default=False)
parser.add_argument('--method', type=str, default='ALAD')

args = parser.parse_args()

if __name__ == "__main__":
    fix_random_seed_as(args.seed)
    get_user_input_clp(args)
    set_template(args)
    train(args)
