from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--speakers', nargs='+')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model_name', type=str)
    
    #for Tmpt and S2G
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--template_length', default=0, type=int)

    #for training from a ckpt
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_pth', default=None, type=str)
    parser.add_argument('--style_layer_norm', action='store_true')
    
    #required
    parser.add_argument('--config_file', default='./config/test.json', type=str)

    # for visualization and test
    parser.add_argument('--audio_file', default=None, type=str)
    parser.add_argument('--text_file', default=None, type=str)
    parser.add_argument('--speaker_names', nargs='+')
    parser.add_argument('--only_face', action='store_true')
    parser.add_argument('--num_sample', default=1, type=int)
    parser.add_argument('--face_model_name', default='s2a_face', type=str)

    parser.add_argument('--face_model_path', default='./checkpoints/face.pth', type=str)
    parser.add_argument('--body_model_name', default='s2a_body_retnet', type=str)
    parser.add_argument('--body_model_path', default='./checkpoints/retnet.pth', type=str)
    parser.add_argument('--infer', action='store_true')

    return parser