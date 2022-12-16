import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import torch,gc

gc.collect()
torch.cuda.empty_cache()

torch.cuda.current_device()
torch.cuda._initialized = True
#解决WIN10无法训练的问题

# torch.backends.cudnn.enabled = False
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    # cudnn.benchmark = True   # DICOM
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:

        fig_path = os.path.join(args.save_path, 'fig')
        x_path = os.path.join(args.save_path, 'x')
        y_path = os.path.join(args.save_path, 'y')
        pred_path = os.path.join(args.save_path, 'pred')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        if not os.path.exists(x_path):
            os.makedirs(x_path)
            print('Create path : {}'.format(x_path))
        if not os.path.exists(y_path):
            os.makedirs(y_path)
            print('Create path : {}'.format(y_path))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
            print('Create path : {}'.format(pred_path))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))


        x_path = os.path.join(args.save_path, 'x')
        y_path = os.path.join(args.save_path, 'y')
        pred_path = os.path.join(args.save_path, 'pred')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        if not os.path.exists(x_path):
            os.makedirs(x_path)
            print('Create path : {}'.format(x_path))
        if not os.path.exists(y_path):
            os.makedirs(y_path)
            print('Create path : {}'.format(y_path))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
            print('Create path : {}'.format(pred_path))

    train_data_loader, test_data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             train_path = args.train_path, 
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers,
                             norm=args.norm, patch_training = args.patch_training)

    solver = Solver(args, train_data_loader, test_data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--train_path', type=str, default='./train_data/train')
    parser.add_argument('--saved_path', type=str, default='./test_data/test/')
    parser.add_argument('--save_path', type=str, default='./save/models/')

    parser.add_argument('--test_patient', type=str, default='LDCT')
    parser.add_argument('--result_fig', action='store_true')

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-1024.0)
    parser.add_argument('--trunc_max', type=float, default=3072.0)

    # parser.add_argument('--norm_range_min', type=float, default=0.0)
    # parser.add_argument('--norm_range_max', type=float, default=1.0)
    # parser.add_argument('--trunc_min', type=float, default=-1024.0)
    # parser.add_argument('--trunc_max', type=float, default=3072.0)

    parser.add_argument('--transform', action='store_true')

    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)  # 10
    parser.add_argument('--patch_size', type=int, default=64)  # 64, make it divisible by 16
    parser.add_argument('--batch_size', type=int, default=4)  # 16

    # logistic
    parser.add_argument('--log_dir', type=str, default='./log/') # tensorboard events writes here
    parser.add_argument('--yaml_path', type=str, default='./config/base.yaml') # config yaml stores here

    # architecture
    parser.add_argument('--model', type=str, default='REDCNN')

    # CNCL
    parser.add_argument('--content_mode', type=str, default='unet')
    parser.add_argument('--noise_mode', type=str, default='unet')
    parser.add_argument('--attn_mode', type=str, default='base')
    parser.add_argument('--norm_mode', type=str, default='bn')
    parser.add_argument('--act_mode', type=str, default='relu')
    parser.add_argument('--fusion_mode', type=str, default='simple')

    # training
    parser.add_argument('--norm', action='store_true') # whether to subtract mean and divide var, true if set
    parser.add_argument('--patch_training', action='store_true') # whether to enable patch_training, true if set
    parser.add_argument('--alter_gan', action='store_true') # whether to enable alterly training in gan, true if set
    parser.add_argument('--discriminator_iters', type=int, default=10) # interval between updating generator and discriminator
    parser.add_argument('--generator_lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=5e-5)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=8655) # for REDCNN
    parser.add_argument('--save_iters', type=int, default=10000)
    parser.add_argument('--val_iters', type=int, default=4000) # 
    parser.add_argument('--test_iters', type=int, default=50000)  # 50000

    parser.add_argument('--lr', type=float, default=1e-4)  # 1e-5

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--multi_gpu', action='store_true')

    args = parser.parse_args()
    main(args)
