import sys
import os
import numpy as np
import cv2

def main(args):

    print('It creates a label file for a whole video.')
    print('it is expected that a frame contains only 1 label information.')
    print('i.e. video with sitting, standing, falling.')
    print('i.e. in the video it is expected to be only a subject.')
    
    
    # open a video
    cap = cv2.VideoCapture(args.fnamein)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
    
        #The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
        #The second argument defines the frame number in range 0.0-1.0
        #cap.set(2,frame_no);

        #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame)+" frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready

        c = cv2.waitKey(0)
        print('c:{}'.format(c))
        if c == 27: break
        if c == 112: #p
            pos_frame = pos_frame - 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            if pos_frame < 0: pos_frame = 0
            
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--fnamein', type=str, default='.', help='videoin')
    parser.add_argument('--fnameout', type=str, default='.', help='save result')

    args = parser.parse_args()

    main(args)
