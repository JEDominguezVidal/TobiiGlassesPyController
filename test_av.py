# live_scene_gaze_and_aoi.py : A demo for video streaming, gaze and synchronized Area of Interest (AoI)
#
# Copyright (C) 2018  Davide De Tommaso
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import cv2
import av
import numpy as np
from tobiiglassesctrl import TobiiGlassesController

# Initial configuration
ipv4_address = "192.168.100.10"

tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)
tobiiglasses.start_streaming()

rtsp_url = "rtsp://%s:8554/live/scene" % ipv4_address
container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
stream = container.streams.video[0]

for frame in container.decode(stream):
    data_gp = tobiiglasses.get_data()['gp']
    data_pts = tobiiglasses.get_data()['pts']
    frame_cv = frame.to_ndarray(format='bgr24')

    if data_gp['ts'] > 0 and data_pts['ts'] > 0:
        # offset = data_gp['ts'] / 1000.0 - data_pts['ts'] / 1000.0  # in milliseconds
        # print('Frame_pts = %f' % float(frame.pts))
        # print('Frame_time = %f' % float(frame.time))
        # print('Data_pts = %f' % float(data_pts['pts']))
        # print('Offset = %f' % float(offset))

        # Overlay gazepoint
        height, width = frame_cv.shape[:2]
        cv2.circle(frame_cv, (int(data_gp['gp'][0] * width), int(data_gp['gp'][1] * height)), 20, (0, 0, 255), 6)
    # Display Stream
    cv2.imshow("Livestream", frame_cv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
tobiiglasses.stop_streaming()
tobiiglasses.close()
