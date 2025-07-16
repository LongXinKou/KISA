import argparse
# video encoder
def main_config():
    parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
    # ====Experiment====
    parser.add_argument('--version', default=1, type=int)
    parser.add_argument('--train_mode', default='vlm', type=str,
                        help='vlm/pred')
    parser.add_argument('--test_mode', default='vlm', type=str,
                        help='vlm/pred')
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--seed', default=42, type=int) 

    # ====Network====
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--visual_representation', default='vvip', type=str,
                        help='liv/r3m/clip/vip/vclip/vr3m/vliv/vvip')
    parser.add_argument('--predictor', default='mlp', type=str,
                        help='mlp/cvae')
    parser.add_argument('--model_path', default="result/vlm/2024-01-11T21:20/vencoder_checkpoint_260.pth.tar", type=str,
                        help='the path for the pretrained model weight')
    parser.add_argument('--pred_model_path', default="result/predictor/kitchen_vliv_v1_001_mlp_100/predictor_checkpoint_100.pth.tar", type=str,
                        help='the path for the pretrained model weight')

    # ====Training Parameters====
    parser.add_argument('--train_path', default="/data/ubuntu/VideoRLCS/dataset/rebuttle/train_dataset.h5", type=str)
    parser.add_argument('--test_path', default="/data/ubuntu/VideoRLCS/dataset/rebuttle/test_dataset.h5", type=str)
    parser.add_argument('--num_iteration',default=400, type=int, help='The maximum Epochs for learn')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch_size for training')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--alpha', default=0.1, type=float, help='loss')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay (default: 1e-2)',)
    parser.add_argument('--decay_steps', type=int, default=2e5)
    parser.add_argument('--save_model_every_n_steps',default=100, type=int, help='The frequency for saving model')

    # ====Netwrok Parameters====
    parser.add_argument('--hidden_size', type=int, default=1024)
    # transformer block
    parser.add_argument('--temporal', type=int, default=1)
    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=2)
    # attention block
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--att_heads', type=int, default=2)
    # cvae block
    parser.add_argument('--latent_dim', type=int, default=32)

    # ====Test Setting====
    parser.add_argument('--save_dir', default='./tmp_test', type=str,
                        help='the path to save the visualization result')
    args = parser.parse_args()
    return args

def retrieval_config():
    parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
    # ====Experiment====
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--seed', default=42, type=int)
    # ====Environment Setting====
    parser.add_argument('--env', default='maniskill2', type=str,
                        help='Environment name')
    parser.add_argument('--env_id', default='StackCube-v0', type=str,
                        help='Task of environment')
    parser.add_argument('--obs_mode', default='rgbd', type=str,
                        help='Mode of observation, including state and rgbd')
    parser.add_argument('--control_mode', default='pd_ee_delta_pose', type=str,
                        help='Mode of robot control')

    parser.add_argument('--visual_representation', default='liv', type=str,
                        help='clip/r3m/liv')
    # ====Data Setting====
    parser.add_argument('--data_dir', default="./Dataset/maniskill2/v1/test/StackCube-v0.h5", type=str)

    args = parser.parse_args()
    return args

def gtscore_config():
    parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--visual_representation', default='vvip', type=str)
    parser.add_argument('--predictor', default='mlp', type=str)
    parser.add_argument('--model_path', default="/data/ubuntu/VideoRLCS/result/vlm/vvip_a01_300/vencoder_checkpoint_280.pth.tar", type=str)
    parser.add_argument('--file_path', default="/data/ubuntu/VideoRLCS/dataset/calvin/test_dataset.h5", type=str)

    # ====Netwrok Parameters====
    parser.add_argument('--hidden_size', type=int, default=1024)
    # transformer block
    parser.add_argument('--temporal', type=int, default=1)
    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=2)

    args = parser.parse_args()
    return args

def test_config():
    parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--visual_representation', default='vvip', type=str)
    parser.add_argument('--model_path', default="result/vlm/vvip_a01_300/vencoder_checkpoint_300.pth.tar", type=str)
    parser.add_argument('--train_path', default="/data/ubuntu/VideoRLCS/dataset/calvin/train_dataset.h5", type=str)
    parser.add_argument('--test_path', default="/data/ubuntu/VideoRLCS/dataset/calvin/test_dataset.h5", type=str)

    # ====Netwrok Parameters====
    parser.add_argument('--hidden_size', type=int, default=1024)
    # transformer block
    parser.add_argument('--temporal', type=int, default=1)
    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=2)

    args = parser.parse_args()
    return args