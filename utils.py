import glob
import json
import os
import platform
import random
import re
import shutil
from datetime import datetime, timedelta

import cv2
import logging
import pandas as pd
import pytesseract
import time
from contextlib import contextmanager
from json import JSONDecodeError

import youtube_dl
from _socket import gaierror
from requests import ConnectionError as RequestsConnectionError
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type, before_sleep_log
from typing import Dict, List, Tuple, Optional

from nba_api.stats.endpoints import playbyplayv2, videoeventsasset
from urllib3.exceptions import HTTPError

logger = logging.getLogger(__name__)

# Each shot type to it's name
prior_shot_type_to_shot_dsc = {
    1: 'JUMP_SHOT',
    2: 'RUNNING_JUMP_SHOT',
    3: 'HOOK_SHOT',
    5: 'LAYUP',
    6: 'DRIVING_LAYUP',
    7: 'DUNK',
    8: 'SLAM_DUNK',
    9: 'DRIVING_DUNK',
    40: 'LAYUP',
    41: 'RUNNING_LAYUP',
    42: 'DRIVING_LAYUP',
    43: 'ALLEY_OOP_LAYUP',
    44: 'REVERSE_LAYUP',
    45: 'JUMP_SHOT',
    46: 'RUNNING_JUMP_SHOT',
    47: 'TURNAROUND_JUMP_SHOT',
    48: 'DUNK',
    49: 'DRIVING_DUNK',
    50: 'RUNNING_DUNK',
    51: 'REVERSE_DUNK',
    52: 'ALLEY_OOP_DUNK',
    55: 'HOOK_SHOT',
    56: 'RUNNING_HOOK_SHOT',
    57: 'DRIVING_HOOK_SHOT',
    58: 'TURNAROUND_HOOK_SHOT',
    63: 'FADEAWAY_JUMPER',
    65: 'JUMP_HOOK_SHOT',
    66: 'JUMP_BANK_SHOT',
    67: 'HOOK_BANK_SHOT',
    71: 'FINGER_ROLL_LAYUP',
    73: 'DRIVING_REVERSE_LAYUP',
    74: 'RUNNING_REVERSE_LAYUP',
    75: 'DRIVING_FINGER_ROLL_LAYUP',
    76: 'RUNNING_FINGER_ROLL_LAYUP',
    77: 'DRIVING_JUMP_SHOT',
    78: 'FLOATING_JUMP_SHOT',
    79: 'PULLUP_JUMP_SHOT',
    80: 'STEP_BACK_JUMP_SHOT',
    81: 'PULLUP_BANK_SHOT',
    82: 'DRIVING_BANK_SHOT',
    83: 'FADEAWAY_BANK_SHOT',
    84: 'RUNNING_BANK_SHOT',
    85: 'TURNAROUND_BANK_SHOT',
    86: 'TURNAROUND_FADEAWAY_SHOT',
    88: 'DRIVING_SLAM_DUNK',
    89: 'REVERSE_SLAM_DUNK',
    90: 'RUNNING_SLAM_DUNK',
    93: 'DRIVING_BANK_HOOK_SHOT',
    94: 'JUMP_BANK_HOOK_SHOT',
    95: 'RUNNING_BANK_HOOK_SHOT',
    96: 'TURNAROUND_BANK_HOOK_SHOT',
    98: 'CUTTING_LAYUP_SHOT',
    99: 'CUTTING_FINGER_ROLL_LAYUP_SHOT',
    100: 'RUNNING_ALLEY_OOP_LAYUP_SHOT',
    101: 'DRIVING_FLOATING_JUMP_SHOT',
    102: 'DRIVING_FLOATING_BANK_JUMP_SHOT',
    103: 'RUNNING_PULL',
    104: 'STEP_BACK_BANK_JUMP_SHOT',
    105: 'TURNAROUND_FADEAWAY_BANK_JUMP_SHOT',
    106: 'RUNNING_ALLEY_OOP_DUNK_SHOT',
    108: 'CUTTING_DUNK_SHOT',
    109: 'DRIVING_REVERSE_DUNK_SHOT',
    110: 'RUNNING_REVERSE_DUNK_SHOT'
}

# Each shot type to it's number of plays throughout the whole NBA-API database
prior_shot_type_histogram = {
    1: 730831,
    2: 11918,
    3: 25977,
    5: 80179,
    6: 97103,
    7: 18542,
    8: 1545,
    9: 10152,
    40: 480,
    41: 29625,
    42: 36213,
    43: 8101,
    44: 13496,
    45: 1181,
    46: 6828,
    47: 28379,
    48: 181,
    49: 3770,
    50: 12209,
    51: 939,
    52: 15174,
    55: 208,
    56: 777,
    57: 10332,
    58: 18482,
    63: 30673,
    65: 1385,
    66: 9190,
    67: 1272,
    71: 4399,
    73: 16346,
    74: 2791,
    75: 32975,
    76: 6354,
    77: 1555,
    78: 41092,
    79: 187947,
    80: 69034,
    81: 811,
    82: 2323,
    83: 573,
    84: 955,
    85: 1481,
    86: 21988,
    88: 639,
    89: 38,
    90: 192,
    93: 1598,
    94: 28,
    95: 29,
    96: 1394,
    98: 35834,
    99: 4628,
    100: 570,
    101: 56258,
    102: 17084,
    103: 8435,
    104: 523,
    105: 3057,
    106: 1497,
    108: 21683,
    109: 331,
    110: 143
}

hook_shot_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'HOOK' in v}
bank_shot_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'BANK' in v}
jump_shot_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'JUMP_SHOT' in v}
layup_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'LAYUP' in v}
dunk_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'DUNK' in v}
cut_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'CUT' in v}
putback_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'TIP_' in v or 'PUTBACK' in v}


class ActionGapManager:
    """ This is due to the NBA API blocking us if we make requests too frequently. It's a cooldown mechanism"""

    def __init__(self, gap=0.6):
        self.gap = gap
        self.last_action_time = None

    def _wait_for_gap(self):
        if self.last_action_time is not None:
            elapsed_time = time.time() - self.last_action_time
            if elapsed_time < self.gap:
                time.sleep(self.gap - elapsed_time)

    def _update_last_action_time(self):
        self.last_action_time = time.time()

    @contextmanager
    def action_gap(self):
        try:
            self._wait_for_gap()
            yield
        finally:
            self._update_last_action_time()


# This is for not overloading the NBA API and getting blocked
nba_api_cooldown = 0.6
gap_manager = ActionGapManager(gap=nba_api_cooldown)


@retry(stop=stop_after_attempt(50), wait=wait_random(min=1, max=2),
       retry=retry_if_exception_type((JSONDecodeError, ConnectionError, gaierror, HTTPError, RequestsConnectionError)),
       reraise=True,
       before_sleep=before_sleep_log(logger, logging.DEBUG))
def get_pbp_data(game_id):
    with gap_manager.action_gap():
        raw_data = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=60 * 5)
    df = raw_data.get_data_frames()[0]
    return df


@retry(stop=stop_after_attempt(50), wait=wait_random(min=1, max=2),
       retry=retry_if_exception_type((JSONDecodeError, ConnectionError, gaierror, HTTPError, RequestsConnectionError)),
       reraise=True,
       before_sleep=before_sleep_log(logger, logging.DEBUG))
def _get_video_event_json_from_api(game_id: str, game_event_id: str) -> Dict:
    with gap_manager.action_gap():
        raw_data = videoeventsasset.VideoEventsAsset(game_id=game_id, game_event_id=str(game_event_id), timeout=5 * 60)
    json = raw_data.get_dict()
    return json


def get_video_event_info(game_id, game_event_id) -> Dict[str, str]:
    video_event_dict = _get_video_event_json_from_api(game_id, game_event_id)
    video_urls = video_event_dict['resultSets']['Meta']['videoUrls']
    playlist = video_event_dict['resultSets']['playlist']
    return {'desc': playlist[0]['dsc'], 'video_url': video_urls[0]['lurl']}


# rf = Roboflow(api_key="8VOyQWRE73L2itfMq1wC")
# project = rf.workspace().project("basketball-players-fy4c2")
# model = project.version(20).model


# def get_bbox(frame):
#     predictions = model.predict(frame, confidence=10, overlap=30).json()['predictions']
#     predictions = [prediction for prediction in predictions
#                    if prediction['class'] == 'Time Remaining']
#     for time_remaining_prediction in predictions:
#         y = math.floor(time_remaining_prediction['y'])
#         x = math.floor(time_remaining_prediction['x'])
#         height = math.ceil(time_remaining_prediction['height'])
#         width = math.ceil(time_remaining_prediction['width'])
#         start_point = int(x - width / 2), int(y + height / 2)
#         end_point = int(x + width / 2), int(y - height / 2)
#         yield start_point, end_point


def cut_video(video_path: str, start_time: str, cut_duration: int, output_path: str,
              new_resolution: Optional[Tuple[int, int]] = None) -> bool:
    if platform.system().lower() == 'windows':
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    elif platform.system().lower() == 'linux':
        pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/pytesseract'
    else:
        raise Exception("I don't know what to do for non windows or linux OS")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_record = int(cut_duration * fps)
    current_frame = 0

    try:
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            # Video has ended, without us recording anything
            return False

        # Get height, width, and channels from the frame.shape tuple
        height, width, channels = frame.shape

        while True:
            # Crop the bottom third of the frame
            crop_img = frame[height - height // 4:height, 0:width]
            # Continue with your image processing steps...
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            blurred = cv2.GaussianBlur(bw, (5, 5), 0)
            text_data = pytesseract.image_to_string(blurred, lang='eng', config='--psm 11')

            # Check if the game clock matches the condition, and we should start recording
            if start_time in text_data:
                # Exit to start recording
                break
            else:
                # Jump 1 second forward
                current_frame += fps
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                # Read next frame
                ret, frame = cap.read()
                if not ret:
                    # Video has ended, without us recording anything
                    return False

        # Initialize the video writer to save the cut video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Specify the resolution for the cut video
        resolution = (width, height) if not new_resolution else new_resolution
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        new_video_current_frame = 0

        while new_video_current_frame < frames_to_record and ret:
            # Resize the frame to the desired resolution before writing it
            frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            # Write the frame to the cut video
            out.write(frame)
            new_video_current_frame += 1
            current_frame += 1
            # Read next frame
            ret, frame = cap.read()

            # # Display the processed frame (optional)
            # cv2.imshow("Processed Frame", frame)
            #
            # # Press 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # We're done recording
        out.release()
        return new_video_current_frame > 0

    finally:
        # Release the video capture and close all windows
        cap.release()
        # cv2.destroyAllWindows()


def add_seconds_to_time(time_str, seconds_to_add):
    # Parse the input time string into a datetime object
    time_format = "%M:%S"
    time_obj = datetime.strptime(time_str, time_format)

    if time_obj.minute != 0:
        # Regular case. There is what to decrease from
        new_time_obj = time_obj + timedelta(seconds=seconds_to_add)
    else:
        # Spacial case. Need to make sure we don't go below 0
        new_time_obj = time_obj + timedelta(seconds=max(seconds_to_add, -time_obj.second))

    # Check if the time is under a minute
    if new_time_obj.minute == 0:
        # Format the result as seconds only (without leading zeros)
        result_time_str = f'{new_time_obj.strftime("%S")}.'
    else:
        # Format the result as minutes and seconds
        result_time_str = new_time_obj.strftime("%M:%S")

    # Remove leading zero from minutes (if present)
    if result_time_str.startswith('0'):
        result_time_str = result_time_str[1:]

    return result_time_str


def get_shots_event_data_from_game_df(df):
    REBOUND_EVENT_TYPE = 4

    time_format = "%M:%S"
    # Add time from last event based on the shot clock
    df['TIME_FROM_PREVIOUS_EVENT'] = \
        df['PCTIMESTRING'].shift(1, fill_value="12:00").apply(lambda x: datetime.strptime(x, time_format)) - \
        df['PCTIMESTRING'].apply(lambda x: datetime.strptime(x, time_format))
    # Remove shots which happens less than 4 secs directly after/before a rebound. These videos likely contains 2 shots.
    indices_to_remove = df[
        # Event is a rebound
        (df['EVENTMSGTYPE'] == REBOUND_EVENT_TYPE) &
        # Event less than 4 seconds after is a shot
        ((df['EVENTMSGTYPE'].shift(1) == 2) & (df['TIME_FROM_PREVIOUS_EVENT'] < pd.Timedelta(seconds=4))) &
        # Event less than 4 seconds before is a shot
        ((df['EVENTMSGTYPE'].shift(-1) <= 2) & (df['TIME_FROM_PREVIOUS_EVENT'].shift(-1) < pd.Timedelta(seconds=4)))
        ].index
    adjacent_indices = [index - 1 for index in indices_to_remove] + [index + 1 for index in indices_to_remove]
    # Remove rows from df based on indices
    df = df.drop(indices_to_remove).drop(adjacent_indices)
    # Remove every play other than a shot
    df = df[df['EVENTMSGTYPE'] <= 2]
    # Remove plays without video
    df = df[df["VIDEO_AVAILABLE_FLAG"] == 1]
    # Remove blocked shots. We don't want them because They'll be harder to classify
    df = df[~(df['HOMEDESCRIPTION'].str.contains('BLOCK') | df['VISITORDESCRIPTION'].str.contains('BLOCK'))]
    # Remove tip and putback shots. They will be hard to classify
    df = df[~df['EVENTMSGACTIONTYPE'].isin(putback_classes.keys())]
    # Create `DESCRIPTION` from either teams column (doesn't matter to us)
    # Makes sure before that we didn't mess up, and have a play wite 2 descriptions
    if df[['VISITORDESCRIPTION', 'HOMEDESCRIPTION']].notna().all(axis=1).any():
        raise ValueError("df has a row where both `VISITORDESCRIPTION` and `HOMEDESCRIPTION` and not None")
    df['DESCRIPTION'] = df['HOMEDESCRIPTION'].fillna(df['VISITORDESCRIPTION'])
    # Make sure that every line has a not-None description
    if not df['DESCRIPTION'].notna().all():
        raise ValueError("df has a row where `DESCRIPTION` is None")
    # Filter out irrelevant data
    shots_event_data = df[
        ['EVENTNUM', 'EVENTMSGACTIONTYPE', 'PERIOD', 'PCTIMESTRING', 'DESCRIPTION', 'EVENTMSGTYPE',
         'VIDEO_AVAILABLE_FLAG']]
    return shots_event_data


def get_event_msg_action(description):
    p = re.compile('(\s{2}|\' )([\w+ ]*)')

    match = p.findall(description)
    if not match:
        return None
    else:
        event_msg_action = re.sub(' ', '_', match[0][1]).upper().rstrip("_").replace("3PT_", "")
        return event_msg_action


def organize_dataset_from_videos_folder(root_dir: str, new_root_dir: str,
                                        video_type_categories: List[str], number_of_videos_per_category: int):
    # Define the new subdirectories
    subdirectories = ['train', 'val', 'test']
    if number_of_videos_per_category % 10 != 0:
        raise Exception("Number has to be a multiply of 10!")

    # Create the new directory structure
    if not os.path.exists(new_root_dir):
        os.makedirs(new_root_dir)
        for subdirectory in subdirectories:
            os.makedirs(os.path.join(new_root_dir, subdirectory))
            for video_type in video_type_categories:
                video_type_dir = os.path.join(new_root_dir, subdirectory, video_type)
                os.makedirs(video_type_dir)

    # Move video files

    for video_type in video_type_categories:
        source_dir = os.path.join(root_dir, video_type)

        video_files = glob.glob(os.path.join(source_dir, "*", "*.mp4"))
        random.shuffle(video_files)

        for subdirectory in subdirectories:
            dest_dir = os.path.join(new_root_dir, subdirectory, video_type)
            if subdirectory == 'train':
                selected_videos = video_files[:int(number_of_videos_per_category * 0.8)]
            elif subdirectory == 'val':
                selected_videos = video_files[
                                  int(number_of_videos_per_category * 0.8):int(number_of_videos_per_category * 0.9)]
            elif subdirectory == 'test':
                selected_videos = video_files[int(number_of_videos_per_category * 0.9):number_of_videos_per_category]
            else:
                raise Exception

            print(f"Coping {video_type} videos for {subdirectory}")
            for video_file in selected_videos:
                source_file = video_file
                dest_file = os.path.join(dest_dir, f"{os.path.basename(os.path.dirname(source_file))}.mp4")
                shutil.copy(source_file, dest_file)

    print("Folder structure and file movement complete.")


def download_video(event_info, info_path, video_path):
    # Save video_event info
    with open(info_path, "w") as outfile:
        json.dump(event_info, outfile)
    # Save video
    ydl_opts = {'outtmpl': video_path.as_posix(), 'quiet': True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([event_info['video_url']])
