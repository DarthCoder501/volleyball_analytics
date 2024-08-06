from pathlib import Path
from typing import List

import cv2
import yaml
from numpy.typing import NDArray
from tqdm import tqdm
from gamestate import GameStateClassifier
from src.backend.app.models.models import Video
from src.ml.game_state_classification.utils import Manager

if __name__ == '__main__':
    config = '/home/masoud/Desktop/projects/volleyball_analytics/conf/ml_models.yaml'
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
    model = GameStateClassifier(cfg=cfg['video_mae']['game_state_3'])
    src = Video.get(1)
    video = src.path

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), "file is not accessible..."

    rally_save_path = '/media/HDD/DATA/volleyball/rallies'
    service_save_path = '/media/HDD/DATA/volleyball/serves'
    mistake_path = '/media/HDD/DATA/volleyball/wrong_annotation'

    width, height, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]

    previous_state = 'no-play'
    current_state = None
    no_play_flag = 0
    state_manager = Manager(fps, width, height, 30)
    pbar = tqdm(list(range(n_frames)))
    for fno in pbar:
        pbar.update(1)
        cap.set(1, fno)
        status, frame = cap.read()
        state_manager.append_frame(frame)

        if state_manager.is_full():
            current_frames, current_fnos = state_manager.get_current_frames_and_fnos()
            current_state = model.predict(current_frames)
            state_manager.set_current_state(current_state)
            pbar.set_description(f"state: {current_state}")
            current = state_manager.current_state
            prev = state_manager.previous_state
            prev_prev = state_manager.before_previous_state

            match current:
                case 'service':
                    if prev in ('no-play', 'service'):
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        state_manager.reset_temp_buffer()
                    elif prev == 'play':
                        # In the middle of `play`, we never get `service` unless the model is wrong.
                        # we save the video to investigate the case.
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        # TODO: Fix write_video and try again.
                        all_labels: List[int] = state_manager.get_labels()
                        all_frames: List[NDArray] = state_manager.get_long_buffer()
                        all_fnos: List[int] = state_manager.get_long_buffer_fno()
                        state_manager.write_video(path=Path(mistake_path),
                                                  width=width,
                                                  height=height,
                                                  fps=fps,
                                                  labels=all_labels,
                                                  long_buffer=all_frames,
                                                  long_buffer_fno=all_fnos,
                                                  draw_label=True)
                        print("Mistake .....")
                        state_manager.reset_temp_buffer()
                        state_manager.reset_long_buffer()
                        # Reset states, keep the current frames, but removing previous frames.
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                case 'play':
                    if prev == 'service':
                        # Save the state buffer as the service video and keep buffering the rest ...
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        all_labels: List[int] = state_manager.get_labels()
                        all_frames: List[NDArray] = state_manager.get_long_buffer()
                        all_fnos: List[int] = state_manager.get_long_buffer_fno()
                        state_manager.write_video(path=Path(service_save_path),
                                                  width=width,
                                                  height=height,
                                                  fps=fps,
                                                  labels=all_labels,
                                                  long_buffer=all_frames,
                                                  long_buffer_fno=all_fnos,
                                                  draw_label=True)

                        print(f"Caught a service... saved in {service_save_path}")

                        state_manager.reset_temp_buffer()
                    elif prev == 'play':
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        state_manager.reset_temp_buffer()
                    elif prev == 'no-play':
                        # TODO: Check this part, not making problems.
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        state_manager.reset_temp_buffer()
                case 'no-play':
                    # Only 2 consecutive "no-play" means the end of rally...
                    if prev in ('service', 'play'):
                        # if we haven't got 2 cons
                        state_manager.keep(current_frames, current_fnos, [current] * len(current_fnos))
                        # store the whole game video from service to end of the game.
                    elif prev == 'no-play':
                        # This is when we have 2 states of 'no-plays', so let's store the video.
                        # if the state before prev is 'service', save it as a service, if `play`,
                        # store it as a `rally`, otherwise keep skipping the video.
                        if prev_prev == 'play':
                            all_labels: List[int] = state_manager.get_labels()
                            all_frames: List[NDArray] = state_manager.get_long_buffer()
                            all_fnos: List[int] = state_manager.get_long_buffer_fno()
                            state_manager.write_video(path=Path(rally_save_path),
                                                      width=width,
                                                      height=height,
                                                      fps=fps,
                                                      labels=all_labels,
                                                      long_buffer=all_frames,
                                                      long_buffer_fno=all_fnos,
                                                      draw_label=True)
                            print(f"Caught a RALLY ... saved in {rally_save_path}")

                        elif prev_prev == 'service':
                            all_labels: List[int] = state_manager.get_labels()
                            all_frames: List[NDArray] = state_manager.get_long_buffer()
                            all_fnos: List[int] = state_manager.get_long_buffer_fno()
                            state_manager.write_video(path=Path(service_save_path),
                                                      width=width,
                                                      height=height,
                                                      fps=fps,
                                                      labels=all_labels,
                                                      long_buffer=all_frames,
                                                      long_buffer_fno=all_fnos,
                                                      draw_label=True)
                            print(f"Caught a SERVICE ... saved in {service_save_path}")
                        else:
                            state_manager.reset_long_buffer()
                    state_manager.reset_temp_buffer()
        else:
            continue
