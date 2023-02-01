#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
# Copyright (c) 2022 ACS research group, Institute of Computer Science, University of Tartu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""The entry point for idcard_face_match which includes the GUI event loop"""

import os
import argparse
from pathlib import Path
import threading
from queue import Queue

from idcard_face_match.main_program_loop import main_program_loop
from idcard_face_match.camera import camera
from smartcard.CardMonitoring import CardMonitor
from idcard_face_match.card_comms import CardWatcher


def parse_arguments() -> argparse.Namespace:
    """parse arguments"""

    def raise_(ex):
        """https://stackoverflow.com/a/8294654/6077951"""
        raise ex

    parser = argparse.ArgumentParser(
        description="Biometric facial verification using authentic ID card facial image",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # not necessary - files are shipped
    #parser.add_argument(
    #    "--online",
    #    action="store_true",
    #    help="Download CRL and CSCA certificates online",
    #)
    # not implemented
    #parser.add_argument(
    #    "--online_check",
    #    action="store_true",
    #    help="Check ID card revocation status online at politsei.ee",
    #)
    parser.add_argument(
        "--output",
        "--o",
        type=lambda x: Path(x) if os.path.isdir(x) else raise_(NotADirectoryError(x)),
        help="Directory to save eMRTD data files",
    )
    parser.add_argument(
        "--camera",
        default=0,
        type=int,
        help="Device ID of the camera to be used (e.g., 0 will use /dev/video0). The default is 0.",
    )
    parser.add_argument(
        "--camera-resolution",
        type=lambda x: list(map(int,x.split('x'))) if len(x.split('x'))==2 else raise_(ValueError(x)),
        default=[],
        help=("Camera input resolution (for 'MJPG' image format).\n"
             "Resolutions supported by camera can be listed using the command:\n"
             "  v4l2-ctl -D -d /dev/video0 --list-formats-ext | grep -A 100 MJPG\n"
             "If no resolution is provided, the default resolution of the camera will be used.")
    )
    parser.add_argument(
        "--camera-rotate",
        type=lambda x: int(x) if int(x) in (0, 90, 180, 270) else raise_(ValueError(x)),
        default=0,
        help="Degrees by which to rotate camera input clockwise (0, 90, 180, 270).",
    )
    parser.add_argument(
        "--camera-size",
        type=lambda x: int(x) if (int(x) > 0 and int(x) <= 100) else raise_(ValueError(x)),
        default=100,
        help="Size (in percentage) of the camera input frame to use (the default is 100 - full frame).",
    )
    parser.add_argument(
        "--screen-width",
        default=540,
        type=int,
        help=("Width component of the resolution for which to draw graphics.\n"
              "The graphics use a fixed widescreen aspect ratio of 16:9 in portrait mode (9:16).\n"
              "The graphics have been tested for resolutions:\n"
              " - 1440 (1440x2560)\n"
              " - 1080 (1080x1920)\n"
              " - 720 (720x1280)\n"
              " - 540 (540x960) - default")
    )

    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Run window in full screen mode. Press 'f' to toggle full screen mode. Press ESC to exit.",
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Show FPS rate on the screen and log detailed timing measurements in console.",
    )

    args = parser.parse_args()

    return args


def main_event_loop():
    """
    Main GUI event loop
    """
    args = parse_arguments()

    first_run = True
    run = True
    q_camera: Queue = Queue()
    q_main: Queue = Queue()
    q_cardmon: Queue = Queue()

    # only a single instance of card monitoring thread must exist
    cardobserver = CardWatcher(q_main, q_camera, q_cardmon)
    cardmonitor = CardMonitor()
    cardmonitor.addObserver(cardobserver)

    threading.Thread(target=camera, args=(q_main, q_camera, args.camera, args.camera_resolution, args.camera_rotate, args.screen_width, args.show_fps, args.fullscreen, args.camera_size), daemon=True).start()

    while True:

        # (re)start the main thread
        if run:
            threading.Thread(target=main_program_loop, args=(q_main, q_camera, q_cardmon, args, first_run), daemon=True).start()
            first_run = False
            run = False

        # read events and act accordingly
        event = q_main.get()
        print("[+] main(): event:", event)
        if event in ("exit"):
            print("[+] main(): received exit event")
            break
        elif event == "-RUN COMPLETE-":
            run = True
        elif event == "-RAISED EXCEPTION-":
            run = True


if __name__ == "__main__":
    main_event_loop()
