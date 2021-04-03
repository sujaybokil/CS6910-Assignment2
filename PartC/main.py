import argparse
from utils import detect_in_video


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    # whether to take input from webcam or use a saved video file
    ap.add_argument("--webcam", help="to view live feed from webcam", type=bool, default=False)
    ap.add_argument("--video_path", help="path to video file", default=False)

    # saved weights and other files to load the model
    ap.add_argument('--weights', help='Path to model weights', type=str, default='mask_weights\\yolov3_mask_last.weights')
    ap.add_argument('--configs', help='Path to model configs',type=str, default='mask_weights\\yolov3_mask.cfg')
    ap.add_argument('--class_names', help='Path to class names file', type=str, default='mask_weights\\obj.names')

    # Setting thresholds
    ap.add_argument('--confidence_thresh', help='Confidence threshold value', default=0.5)
    ap.add_argument('--nms_thresh', help='Confidence threshold value', default=0.4)

    args = vars(ap.parse_args())

    weights, cfg, classes_fpath = args['weights'], args['configs'], args['class_names']

    confidence_threshold = args['confidence_thresh']
    nms_threshold = args['nms_thresh']

    webcam = args['webcam']
    video_path = args['video_path']

    detect_in_video(video_path, webcam, weights, cfg, classes_fpath, confidence_threshold, nms_threshold)