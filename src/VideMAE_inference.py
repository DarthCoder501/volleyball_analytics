import yaml

from src.ml.video_mae.game_state.gamestate import GameStateClassifier
from src.ml.video_mae.game_state.utils import annotate_service
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("arguments for running volleyball action detection on videos...")
    parser.add_argument(

        "--model_cfg",
        type=str,
        default='/home/masoud/Desktop/projects/volleyball_analytics/conf/ml_models.yaml'
    )

    parser.add_argument(
        '--video_path',
        type=str,
        default="/home/masoud/Desktop/projects/volleyball_analytics/data/raw/videos/train/22.mp4"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="/home/masoud/Desktop/projects/volleyball_analytics/runs/inference/game_state"
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=30
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = yaml.load(open(args.model_cfg), Loader=yaml.SafeLoader)
    state_detector = GameStateClassifier(cfg=cfg['video_mae']['game_state_3'])
    annotate_service(serve_detection_model=state_detector, video_path=args.video_path, output_path=args.output_path,
                     buffer_size=args.buffer_size)
