import re
from datetime import datetime, timedelta

import cv2
import logging
import pytesseract
import time
from contextlib import contextmanager
from json import JSONDecodeError
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type, before_sleep_log
from typing import Dict

from nba_api.stats.endpoints import playbyplayv2, videoeventsasset

logger = logging.getLogger(__name__)

prior_shot_type_to_shot_dsc = {
    0: 'NO_SHOT',
    1: 'JUMP_SHOT',
    2: 'RUNNING_JUMP_SHOT',
    3: 'HOOK_SHOT',
    4: 'TIP_SHOT',
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
    49: 'DRIVING_DUNK',
    50: 'RUNNING_DUNK',
    51: 'REVERSE_DUNK',
    52: 'ALLEY_OOP_DUNK',
    54: 'RUNNING_TIP_SHOT',
    56: 'RUNNING_HOOK_SHOT',
    57: 'DRIVING_HOOK_SHOT',
    58: 'TURNAROUND_HOOK_SHOT',
    63: 'FADEAWAY_JUMPER',
    65: 'JUMP_HOOK_SHOT',
    66: 'JUMP_BANK_SHOT',
    67: 'HOOK_BANK_SHOT',
    71: 'FINGER_ROLL_LAYUP',
    72: 'PUTBACK_LAYUP',
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
    85: 'TURNAROUND_BANK_SHOT',
    86: 'TURNAROUND_FADEAWAY',
    87: 'PUTBACK_DUNK',
    93: 'DRIVING_BANK_HOOK_SHOT',
    97: 'TIP_LAYUP_SHOT',
    98: 'CUTTING_LAYUP_SHOT',
    99: 'CUTTING_FINGER_ROLL_LAYUP_SHOT',
    100: 'RUNNING_ALLEY_OOP_LAYUP_SHOT',
    101: 'DRIVING_FLOATING_JUMP_SHOT',
    102: 'DRIVING_FLOATING_BANK_JUMP_SHOT',
    103: 'RUNNING_PULL',
    105: 'TURNAROUND_FADEAWAY_BANK_JUMP_SHOT',
    106: 'RUNNING_ALLEY_OOP_DUNK_SHOT',
    107: 'TIP_DUNK_SHOT',
    108: 'CUTTING_DUNK_SHOT'
}

hook_shot_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'HOOK_SHOT' in v}
jump_shot_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'JUMP_SHOT' in v}
layup_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'LAYUP' in v}
dunk_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'DUNK' in v}
putback_classes = {k: v for k, v in prior_shot_type_to_shot_dsc.items() if 'TIP_' in v or 'PUTBACK' in v}


class ActionGapManager:
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


def get_pbp_data(game_id):
    with gap_manager.action_gap():
        raw_data = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=60 * 5)
    df = raw_data.get_data_frames()[0]
    return df


@retry(stop=stop_after_attempt(50), wait=wait_random(min=1, max=2),
       retry=retry_if_exception_type((JSONDecodeError, ConnectionError)), reraise=True,
       before_sleep=before_sleep_log(logger, logging.DEBUG))
def get_video_event_dict(game_id: str, game_event_id: str) -> Dict:
    with gap_manager.action_gap():
        raw_data = videoeventsasset.VideoEventsAsset(game_id=game_id, game_event_id=str(game_event_id), timeout=5 * 60)
    json = raw_data.get_dict()
    return json


def cut_video(video_path: str, start_time: str, cut_duration: int, output_path: str) -> bool:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    try:
        # Variables to track video recording
        recording = False
        frames_to_record = int(cut_duration * fps)
        current_frames = 0
        new_video_current_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not recording:
                if not ret:
                    # Video has ended, without us recording anything
                    return False

                # Check text condition every 1 second
                if current_frames % fps == 0:
                    # Get height, width, and channels from the frame.shape tuple
                    height, width, channels = frame.shape
                    # Crop the bottom third of the frame
                    crop_img = frame[height - height // 4:height, 0:width]

                    # Continue with your image processing steps...
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    blurred = cv2.GaussianBlur(bw, (5, 5), 0)
                    text_data = pytesseract.image_to_string(blurred, lang='eng', config='--psm 11')

                    # Check if the condition is met and we should start recording
                    if start_time in text_data:
                        # Set recording
                        recording = True
                        # Initialize the video writer to save the cut video
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            else:  # Recording
                # Check if the required number of frames have been recorded
                if new_video_current_frames > frames_to_record or not ret:
                    # We're done recording
                    out.release()
                    return True
                # Write the frame to the cut video
                out.write(frame)
                new_video_current_frames += 1

            current_frames += 1

            # # Display the processed frame (optional)
            # cv2.imshow("Processed Frame", blurred)
            #
            # # Press 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()


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
    # Remove every play other than a shot
    df = df[df['EVENTMSGTYPE'] <= 2]
    # Remove plays without video
    df = df[df["VIDEO_AVAILABLE_FLAG"] == 1]
    # Remove blocked shots. We don't want them because They'll be harder to classify
    df = df[~(df['HOMEDESCRIPTION'].str.contains('BLOCK') | df['VISITORDESCRIPTION'].str.contains('BLOCK'))]
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


def get_event_msg_action(description, dsc_from_playlist):
    p = re.compile('(\s{2}|\' )([\w+ ]*)')

    if dsc_from_playlist != description:
        # If those are different for reason other than the BLOCK information addition, we want to know
        raise ValueError(f"{dsc_from_playlist} is different that {description}")
    match = p.findall(description)
    if not match:
        return None
    else:
        event_msg_action = re.sub(' ', '_', match[0][1]).upper().rstrip("_").replace("3PT_", "")
        return event_msg_action
