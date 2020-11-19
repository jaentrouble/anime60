import os
import ffmpeg
from pathlib import Path
inputdir = Path('D:/Lab/anime60/data/2160/temp')
outputdir = Path('D:/Lab/anime60/data/1080/ani_like')
kwargs={
    'vcodec' : 'h264_nvenc',
    'rc:v' : 'vbr_hq',
    'cq:v' : '18',
    'video_bitrate' : '30M',
    'profile:v' : 'high',
    'preset' : 'slow',
    's' : '1920x1080'
}
for f in os.listdir(inputdir):


    (
        ffmpeg
        .input(str(inputdir/f))
        .video
        .output(str(outputdir/f'{Path(f).stem}_1080.mp4'),
                **kwargs)
        .run()
    )
