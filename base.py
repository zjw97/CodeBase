import cv2
from tools import putImgToOne
from functools import partial

def onTrackBarSlide(caps, pos):
    for k, v in caps.items():
        v.set(cv2.CAP_PROP_POS_FRAMES, pos)

if __name__ == "__main__":
    video_list = {
        # "parsing_front_seg_pred":"/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_front_seg_pred/matting/hhhhhhhh20210927_182720_parsing_front_seg_pred_S3E32_WiRNN_W416H224.mp4",
        # "parsing_wo_non_local": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_wo_non_local/matting/hhhhhhhh20210927_182720_parsing_wo_non_local_S3E32_WiRNN_W416H224.mp4",

        "parsing_wo_non_local": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_wo_non_local/matting/TV_CAM_20220214_164618_parsing_wo_non_local_S3E32_WiRNN_W416H224.mp4",
        "parsing_front_seg_pred_add_conv_epoch-1": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_front_seg_pred_add_conv/matting/TV_CAM_20220214_164618_parsing_front_seg_pred_add_conv_S3E32_WiRNN_W416H224_epoch-1.mp4",
        "parsing_front_seg_pred_add_conv_epoch-5": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_front_seg_pred_add_conv/matting/TV_CAM_20220214_164618_parsing_front_seg_pred_add_conv_S3E32_WiRNN_W416H224_epoch-5.mp4"

        # "parsing_front_seg_pred": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_front_seg_pred/matting/Wed_Dec_29_21_15_56_2021_ORIG_parsing_front_seg_pred_S3E32_WiRNN_W416H224.mp4",
        # "parsing_wo_non_local": "/home/zjw/REMO/RobustVideoMatting-master/video_prediction/parsing_wo_non_local/matting/Wed_Dec_29_21_15_56_2021_ORIG_parsing_wo_non_local_S3E32_WiRNN_W416H224.mp4",
    }
    cv2.namedWindow("compare")
    for k, v in video_list.items():
        video_list[k] = cv2.VideoCapture(v)
        frame_count = int(video_list[k].get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("time", "compare", 0, frame_count, partial(onTrackBarSlide, video_list))
    pos = 0

    while True:
        img_list = []
        for k, v in video_list.items():
            pos = int(video_list[k].get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("time", "compare", pos)
        for k, v in video_list.items():
            ret, image = v.read()
            if ret:
                cv2.putText(image, k, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                img_list.append(image)

        merged_image = putImgToOne(img_list)
        cv2.imshow("compare", merged_image)
        key = cv2.waitKey(40)
        if key == ord("q"):
            exit()
        elif key == ord(" "):
            key = cv2.waitKey()