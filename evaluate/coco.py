"""
coco整体格式
{
    "info": info,  # "info" -> dict
    "licenses": [license],  # "licenses" -> list
    "images": [image],   # "images" -> list
    "annotations": [annotation],  # "annotations" -> list
    "categories": [category]  # "categories" -> list
}

"info":{
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str,
    "date_created": datetime,
}

"license":{
    "id": int,
    "name": str,
    "url": str,
}

"image":{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}

"annotation":{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
    #     "keypoints": [x1,y1,v1,...],  # object_keypoint 注释
    #     "num_keypoints": int,
}

"category":{
    "id": int,
    "name": str,
    "supercategory": str,
    # "keypoints": [str], # 这两个字段也是只有object_keypoint文件中才有的
    # "skeleton": [edge]
}


"info":{
	"description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
	"url":"http:\/\/mscoco.org",
	"version":"1.0","year":2014,
	"contributor":"Microsoft COCO group",
	"date_created":"2015-01-27 09:11:52.357475"
},

"license": {
	"url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
	"id":1,
	"name":"Attribution-NonCommercial-ShareAlike License"
},

{
	"license":3,
	"file_name":"COCO_val2014_000000391895.jpg",
	"coco_url":"http:\/\/mscoco.org\/images\/391895",
	"height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
	"flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
	"id":391895
},

{
	"segmentation": [[510.66,423.01,511.72,420.03,510.45......]],
	"area": 702.1057499999998,
	"iscrowd": 0,
	"image_id": 289343,
	"bbox": [473.07,395.93,38.65,28.67],
	"category_id": 18,
	"id": 1768
	# "num_keypoints": 10,  # 下面两个字段只在object_keypoint文件中才有
	# "keypoints": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,309,1,177,320,2,191,398...],
},



{
	"supercategory": "person",
	"id": 1,
	"name": "person"
	"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder", \  # 只有object_keypoint中才有
	            "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist", \
	            "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
},

# 用pycocotools计算mAP需要的推理数据格式
[{
    "image_id": int,
    "category_id": int,
    "bbox": [ctr_x, ctr_y, width, height],
    "score"
}]
# 保存到json文件
"""