# Importing necesssary libraries
import argparse
from utils import detect_in_video, detect_in_image


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    # whether to take input from webcam or use a saved image/video file
    ap.add_argument("--image_path", help="Path to input image", default=None)
    ap.add_argument("--webcam", help="To view live feed from webcam", type=bool, default=False)
    ap.add_argument("--video_path", help="Path to video file", default=False)

    # saved weights and other files to load the model
    ap.add_argument('--weights', help='Path to model weights', type=str, default='mask_weights\\yolov3_mask_last.weights')
    ap.add_argument('--configs', help='Path to model configs',type=str, default='mask_weights\\yolov3_mask.cfg')
    ap.add_argument('--class_names', help='Path to class names file', type=str, default='mask_weights\\obj.names')

    # Setting thresholds
    ap.add_argument('--confidence_thresh', help='Confidence threshold value', default=0.5)
    ap.add_argument('--nms_thresh', help='NMS threshold value', default=0.4)

    args = vars(ap.parse_args())

    image_path =  args['image_path']
    webcam = args['webcam']
    video_path = args['video_path']
    weights, cfg, classes_fpath = args['weights'], args['configs'], args['class_names']

    confidence_threshold = args['confidence_thresh']
    nms_threshold = args['nms_thresh']

    if image_path is not None:
        detect_in_image(image_path, weights, cfg, classes_fpath, confidence_threshold, nms_threshold)

    elif webcam or video_path is not None:
        detect_in_video(video_path, webcam, weights, cfg, classes_fpath, confidence_threshold, nms_threshold)

    else:
        print("Error, no argument specified!!")