'''
USAGE:
python hog_detector_vid.py --input ../input/video1.mp4 --output ../outputs/video1_slow.mp4 --speed slow
python hog_detector_vid.py --input ../input/video1.mp4 --output ../outputs/video1_fast.mp4 --speed fast
'''
import cv2 
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='../input/video1.mp4', help='path to the inout video file')
    parser.add_argument('-o', '--output', required=True, help='path to save the output video file')
    parser.add_argument('-s', '--speed', default='yes', choices=['fast', 'slow'],help='whether to use fast or slow detector')
    args = vars(parser.parse_args())
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(args['input'])
    if (cap.isOpened() == False):
        print('Error while trying to open video. Please check again...')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if frame_width < 400: 
            frame_width = 400
            ratio = frame_width / float(frame_width) 
            frame_height = int(frame_width * ratio)
    out = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
    frame_count = 0
    total_fps = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            start_time = time.time()
            frame = cv2.resize(frame, (frame_width, frame_height))
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args['speed'] == 'fast':
                rects, weights = hog.detectMultiScale(img_gray, padding=(4, 4), scale=1.02)
            elif args['speed'] == 'slow':
                rects, weights = hog.detectMultiScale(img_gray, winStride=(4, 4), padding=(4, 4), scale=1.02)
            total_count = 0
            for i, (x, y, w, h) in enumerate(rects):
                if weights[i] < 0.13:
                    continue
                elif weights[i] < 0.3 and weights[i] > 0.13:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    total_count += 1
                if weights[i] < 0.7 and weights[i] > 0.3:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
                    total_count += 1
                if weights[i] > 0.7:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    total_count += 1
            cv2.putText(frame, 'High confidence', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Moderate confidence', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 122, 255), 2)
            cv2.putText(frame, 'Low confidence', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Total Count {total_count}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_write_name = args['input'].split('/')[-1].split('.')[0]
            cv2.imwrite(f"../outputs/frames/{args['speed']}_{frame_write_name}_{frame_count}.jpg", frame)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            cv2.imshow("Preview", frame)
            out.write(frame)
            wait_time = max(1, int(fps/4))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        else:
            break

    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}" )
    cap.release()
    cv2.destroyAllWindows()


main()





