'''
###########################################################################
Import libraries
###########################################################################
'''


try:
    from detectron2.config import get_cfg  
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git'])

#	Import some common libraries
import os, sys, json, cv2, shutil, datetime, time
import numpy as np; import geopandas as gpd
import matplotlib.image as mpimg


from pathlib import Path
from pandas import json_normalize
from shapely.geometry import Polygon, Point
from copy import deepcopy

#	Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


#	Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
# from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.structures import Boxes, BoxMode, pairwise_iou
# from detectron2.export import export_onnx_model#, export_caffe2_model
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

#	Import test time augmentation wrapper
from detectron2.modeling import DatasetMapperTTA, GeneralizedRCNNWithTTA

import detectron2.data.transforms as T

from detectron2.export.flatten import TracingAdapter

#	Import pandarallel for parallelising process across multiple CPU cores
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#	Set the recursion limit 
sys.setrecursionlimit(10**5)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction;

from sahi.utils.detectron2 import export_cfg_as_yaml

def create_instances_json(jsonfile, img_, img_type):
		'''
		Create a json file for all the images to be used 
		in the prediction. 
		It will have the same format if exported from CVAT in COCO format
		'''
		try:
				with open(TRAIN_JSON, "r+") as JSON:
						trainJSON = json.load(JSON)
		except:
				with open(PRED_JSON, "r+") as JSON:
						trainJSON = json.load(JSON)			

		all_files = list(Path(img_).glob(f'*.{img_type}'))
		# print(all_files, img_, img_type)

		print('File 0', os.path.basename(all_files[0]))

		instances_json = {
				"licenses": trainJSON['licenses'],
				"info": trainJSON['info'], 
				"categories": trainJSON['categories'], 
				"images": [],
				"annotations": []}

		def setImgDict(_file_, img_id):
				'''
				Create a dictionary with the image
				metadata in COCO format
				'''
				try:
						im = mpimg.imread(_file_)
						height = im.shape[0]
						width = im.shape[1]

						imgDict = {"id" : img_id,
						"width" : int(width),
						"height" : int(height),
						"file_name" : os.path.basename(_file_),
						"license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
				except(OSError):
						print('Truncate error')
						imgDict = None
				return imgDict
		
		for idx, file_ in enumerate(all_files):
				item = setImgDict(file_,idx+1)
				if item is not None:
						instances_json['images'].append(item)

		with open(jsonfile, 'w') as JSON:
			json.dump(instances_json, JSON)#, indent=2)
			
def polygon_area(x, y):
		'''
		From masks.py # Using the shoelace formula
		https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
		'''
		return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def area(polygons):
		'''
		From masks.py
		Calculates the area of a polygon
		'''
		area = []
		# print(polygons)
		for polygons_per_instance in polygons:
			area_per_instance = 0
			for p in [polygons_per_instance]:
				area_per_instance += polygon_area(p[0::2], p[1::2])
			area.append(area_per_instance)
		return area

def mask_to_polygons(mask, img_width, img_height):
		'''	
		From visualizer.py			
		cv2.RETR_CCOMP flag retrieves all the contours and arranges 
		them to a 2-level hierarchy. External contours (boundary)
		of the object are placed in hierarchy-1. Internal contours
		(holes) are placed in hierarchy-2. cv2.CHAIN_APPROX_NONE 
		flag gets vertices of polygons from contours.
		'''
		# some versions of cv2 does not support incontiguous arr
		mask = np.ascontiguousarray(mask)  
		res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		hierarchy = res[-1]
		if hierarchy is None:  # if empty mask return empty list
			return []

		res = res[-2]
		res = [x.flatten() for x in res]
		# 	These coordinates from OpenCV are integers in range 
		#	[0, W-1 or H-1]. We add 0.5 to turn them into 
		#	real-value coordinate space. A better solution would
		#	be to first +0.5 and then dilate the returned polygon by 0.5.
		res = [x + 0.5 for x in res if len(x) >= 6]

		#	CONVERT FROM ABSOLUTE FLOATING POINT COORDINATES TO RELATIVE COORDINATES ADDED BY ABDULLAH
		#	ASSUMING THAT THE IMAGE IS A SQUARE I.E. 416X416 !!!!!
		# res = [np.divide(x, img_width) for x in res]
		return res

def saveToJSON_box():
		'''
		Save the metadata of the inferred objects to a JSON file
		in YOLO and COCO formats
		'''
		#	check that a folder to store prediction geojsons exists
		if not os.path.exists(PRED_JSON_FOLDER):
				os.makedirs(PRED_JSON_FOLDER)	

		shutil.copyfile(PRED_JSON, f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_cocofmt.json')	

		#	create predictions file for YOLO format
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', 'w') as JSON:
				temp = []
				for idx, d in enumerate( pred_list ):
						temp.append({})						
				json.dump(temp, JSON, indent=2)

		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_cocofmt.json', "r+") as JSON:
				bareJSON = json.load(JSON)

		pred_imgs = bareJSON["images"]
		PredImgNames = [ x["file_name"] for x in pred_imgs ]
		classid = bareJSON["categories"][0]["id"]
		classname = bareJSON["categories"][0]["name"]

		#	read the json files of the coco
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_cocofmt.json', "r+") as JSONcoco:
				tempCOCO = json.load(JSONcoco)

		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', "r+") as JSONyolo:
				tempYOLO = json.load(JSONyolo)

		count = 1	;	dropFrames = []

		#	loop through the images  
		#	TRY MAP OR STARMAP INSTEAD OF FOR LOOP
		no_images = len(pred_list)	
		for idx, d in enumerate( pred_list ):
				# d = pred_list[idx]  	

				#	read the image
				im = cv2.imread(d["file_name"])

				#	get the predicted objects
				outputs = predictor(im)

				#	Get image filename, image id and list of annotations
				filename_ = d["file_name"]
				image_id = d["image_id"]

				#	bring over the outputs from GPU to CPU
				# print(f'pred_list: {pred_list}')
				outputs_cpu = outputs["instances"].to("cpu")

				# print(outputs_cpu)
				# print(t)

				#	NOTE: THE BOXES ARE IN ABSOLUTE COORDINATES --> CONVERT TO RELATIVE COORDINATES 
				#	no need to convert the format of the coordinates in the boxes from XYXY to XYWH
				boxesXYXY = outputs_cpu.pred_boxes.tensor.tolist() if outputs_cpu.has("pred_boxes") else None
				#	well actually I do for uploading back to CVAT BUT not for conversion to shapefile
				boxesXYWH = BoxMode.convert(outputs_cpu.pred_boxes.tensor.numpy(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist() if outputs_cpu.has("pred_boxes") else None

				scores = outputs_cpu.scores.tolist() if outputs_cpu.has("scores") else None

				#	size of image predicted on
				width = tempCOCO["images"][idx]["width"]
				height = tempCOCO["images"][idx]["height"]
				divisor = np.max([width, height])

				#	This 'if' statement would apply to SEGMENTATION predictions
				if outputs_cpu.has("pred_masks"):										
						areas = []
						#	convert input to an array
						masks = np.asarray(outputs_cpu.pred_masks.tolist())
						# print(masks.shape)

						if len(masks.shape) < 3: 
								#	handle no predictions
								_masks_ = None
								seg_polygons = None
						else: 
								#	shape changes from 2d-array to 3d-array
								assert masks.shape[1] != 2, masks.shape
								assert masks.shape == masks.shape, (height, width)
								_masks_ = masks.astype("uint8")

						if _masks_ is not None and None not in _masks_:
								#	convert the segmentation masks to polygons in the function
								# seg_polygons = [];	#area = []
								# for val in _masks_:
								# 		print(val)
								# 		print(mask_to_polygons(val))
								# 		seg_polygons.append( list( mask_to_polygons(val) ) )
								# 		# areas.append(area(mask_to_polygons(val)))

								# seg_polygons = [ list(mask_to_polygons(x)) for x in _masks_]
								seg_polygons = [ list(mask_to_polygons(x, width, height)) for x in _masks_]

								#	calculate the area of each polygon (polygons should be in relative coordinates)
								areas = [ area(x) for x in seg_polygons ]

				elif outputs_cpu.has("pred_masks") is False:
						_masks_ = None
						seg_polygons = None
						#	calculate the area of each polygon (polygons should be in relative coordinates)
						# print(outputs_cpu.keys())
						areas = [ round( x[2] * x[3]  ,3) for x in boxesXYWH ]
						print('areas',areas)

				no_predictions = len(scores)

				#	if the filenames match
				# if PredImgNames[idx] == tempCOCO["images"][idx]["file_name"] and _masks_ is not None:
				if PredImgNames[idx] == tempCOCO["images"][idx]["file_name"] and boxesXYWH is not None:
						#	check the overlap of each prediction's boxes
						#	 if overlap and image id is not the same then merge

						# overlapBoxes = []
						# unionPolygons = []
						#	save the predictions to COCO and YOLO JSONS
						try:
								print(f"Updating: {idx}/{no_images}")

								#	loop through all the predicted objects for the image
								# for jdx, ann in enumerate( seg_polygons ):
								for jdx, ann in enumerate( boxesXYXY ):
										if seg_polygons is None:
												seg_poly = []
												obj_area = areas[jdx]
										elif seg_polygons is not None:
												# print('ann[0]', type(ann[0]))
												# print('ann',ann)
												# seg_poly = ann[0].tolist()
												seg_poly = [ ann[0].tolist() ]
												obj_area = areas[jdx][0]
												# print('seg_poly', type(seg_poly))


										# _box_ = boxesXYXY[jdx]
										# _box_ = [x/divisor for x in boxesXYXY[jdx]]
										_box_ = [x/divisor for x in ann]
										# print('_box_', type(_box_))
										# print('obj_area', type(obj_area))
										segmentCOCO = {
												"id" : count,
												"image_id" : image_id,
												"category_id" : 1,		
												#	THE ABOVE assumes one class only per dataset
												#	PLEASE CHECK IN THE JSON FILE DOWNLOADED FROM CVAT
												"segmentation" : seg_poly,
												"area" : obj_area,
												"bbox" : _box_, 
												"iscrowd": 0, 
												"attributes": {"occluded": False}
												}
										count += 1
										tempCOCO["annotations"].append( segmentCOCO )
										# print('segmentCOCO', type(segmentCOCO))
										# print('tempCOCO', type(tempCOCO))

								obj = []
								segmentYOLO = {"frame_id" : image_id, "filename" : filename_, #	full path to img file
										"objects" : [] }

								#	loop through all the predicted objects for the image
								# for jdx, ann in enumerate( seg_polygons ):
								for jdx, ann in enumerate( boxesXYXY ):
										if seg_polygons is None:
												seg_poly = []
										elif seg_polygons is not None:
												seg_poly = ann[0].tolist()
										YOLOsegments = {
												"class_id" : classid,
												"name" : classname,
												"relative_coordinates" : {
														"img_width": width,
														"img_height": height,
														"center_x" : np.divide(boxesXYWH[jdx][0] + boxesXYWH[jdx][2]/2, divisor),
														"center_y" : np.divide(boxesXYWH[jdx][1] + boxesXYWH[jdx][3]/2, divisor),
														"width" : np.divide(boxesXYWH[jdx][2], divisor),
														"height" : np.divide(boxesXYWH[jdx][3], divisor),
														"segmentation" : seg_poly},
												"confidence" : scores[jdx],
												"frame_id" : image_id}									
										obj.append( YOLOsegments )							
										segmentYOLO["objects"] = obj							
								tempYOLO[idx] = segmentYOLO#obj

						except(IndexError):
								print('IndexError',idx)

				#	if there are no objects detected
				#	should probably change this to use a different list
				# elif _masks_ is None:
				elif boxesXYWH is None:
						print('None')
						dropFrames.append(idx)

		#	open the predictions json and save the predictions in COCO format
		#	save the predictions in YOLO format too.
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_cocofmt.json', 'w') as cocoJSON:												
				json.dump(tempCOCO, cocoJSON, indent=2)

		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', 'w') as yoloJSON:	
				json.dump(tempYOLO, yoloJSON, indent=2)

		#	open the yolo predictions and drop frames/dicts with no predictions		
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', "r+") as JSONread:
				readonly = json.load(JSONread)

		#	start dropping items from the back to try avoid IndexError
		dropFrames = dropFrames[::-1]
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', 'w') as JSONwrite:
				for idx in dropFrames:
						readonly.pop(idx)
				json.dump(readonly, JSONwrite, indent=2)

		print('saved JSONS')

def pred_to_georef():
		'''
		Save the inferred/predicted objects/segmentations to a geojson
		'''
		print(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json')
		print('Begin conversion', datetime.now())
		#	open predictions saved in YOLO format
		with open(f'{PRED_JSON_FOLDER}/{DATA_TYPE}_predictions_yolofmt.json', "r+") as JSON:
				yoloJSON = json.load(JSON)

		print('Begin JSON normalization', datetime.now())
		#	json_normalize is super for flattening json. record_path = what
		#	level defines rows and meta specifies the higher level rows to keep
		results_df = json_normalize(data=yoloJSON, record_path=['objects'], meta=['filename'])
		print('JSON normalized', datetime.now())


		def get_img_dims(row):
				'''
				Return the dimensions/shape of an image.
				Note: tile sizes may vary due to cropping at edges
				'''
				if type(row["filename"]) == str:
						height = row['relative_coordinates.img_height']
						width = row['relative_coordinates.img_width']
						return [height, width]
				else:
						print('None')
						return [None, None]

		results_df['dims'] = results_df.parallel_apply(get_img_dims, axis = 1)
		#	split the dims into separate columns
		print('obtained image dimensions',datetime.now())
		results_df['height'] = results_df['dims'].parallel_apply(lambda h: h[0])
		print('height dimension retrieved', datetime.now())
		results_df['width'] = results_df['dims'].parallel_apply(lambda h: h[1])
		print('width dimension retrieved', datetime.now())

		def get_affine_txt(row, ext=PREDICTIONS_IMG_WORLDFILE):
				'''
				Helper to open the 6 line world file stored as side-car files 
				for geo imagery. https://en.wikipedia.org/wiki/World_file
				World files can be used to extract 6 elements (a-f) of the 
				3x3 affine matrix https://github.com/sgillies/affine
				'''
				try:
						world_file = row["filename"][:-3] + ext
						# wm = np.loadtxt(world_file, delimiter = '\n')
						wm = np.genfromtxt(world_file, delimiter = '\n')
						'''
						lines:params 1:a, 2:d, 3:b, 4:e, 5:c, 6:f
						1: x resolution, 				  2: amount of translation, 
						3: amount of rotation, 			  4: negative of the y resolution, 
						5: xMin coordinate( upper left ), 6: yMax coordinate (upper left)
						'''
						afmat = wm[0], wm[2], wm[4], wm[1], wm[3], wm[5] 
						afmat = np.append(afmat, [0.0, 0.0, 1.0]) # We must pad with 0,0,1 to get 3x3 mat
						return afmat
				except(TypeError):						
						return None

		def MinMax(row):
				'''
				BOUNDING BOX 
				this function is designed to use the affine transform to convert xy pairs to geocoordinates.
				Takes: a row of a dataframe that contains the affine matrix('affine_elem') and the coordinates of
				e.g. a bounding box or any other polygon as a sequence of xy pairs. e.g. a COCO bounding box in XYmin_XYmax format            
				'''
				if row['affine_elem'] is not None:
						affine_mat = row['affine_elem'].reshape(3,3)

						try:
								x_ords = row['bbox'][::2]				#	Every other element starting from first - these are the X px coords
								y_ords = row['bbox'][1::2]				#	Every other element starting from second - these are the y px coords
						except(KeyError):
								try:
										# time.sleep(10)
										print('POLYGON MIN MAX')
										x_ords = row['poly'][::2]		#	Segmentation polygon
										y_ords = row['poly'][1::2]
								except(KeyError):
										# time.sleep(10)
										print('AFFINE MIN MAX')
										x_ords = row['affinebbox'][::2]	#	Affine matrix bounding box
										y_ords = row['affinebbox'][1::2]

						print('x_ords',x_ords, 'y_ords',y_ords)
						pad_ones = np.ones(y_ords.shape[0])				#	padding so we get a 3 x num xy-paris matrix for our matmul
						xy_mat = np.vstack([x_ords, y_ords, pad_ones])	#	Add a row of ones for matmul
						geo_pairs = np.matmul(affine_mat, xy_mat)
						#	matmul: https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html#coordinate-transformation

						minx = np.min(geo_pairs[0,:])	#	lowest value in the row of x values
						maxx = np.max(geo_pairs[0,:])
						miny = np.min(geo_pairs[1,:])
						maxy = np.max(geo_pairs[1,:])			  
						#	now form up the pairs for polygons - preferred order anti-clockwise from lowerleft
						return [minx, maxy, maxx, miny]
				else:
						return None

		def calc_geoobj_affine(row):
				'''
				BOUNDING BOX
				this function is designed to use the affine transform to convert xy pairs to geocoordinates
				Eventual goal is to apply this to segmentation masks too.
				Takes: a row of a dataframe that contains the affine matrix('affine_elem') and the coordinates of
				e.g. a bounding box or any other polygon as a sequence of xy pairs. e.g. a COCO bounding box in XYmin_XYmax format            
				''' 
				# print('Inside the def calc_geoobj_affine function')
				if row['affine_elem'] is not None:
						# print('affine_elem is not None')
						affine_mat = row['affine_elem'].reshape(3,3)						
						try:
								# print('try bbox')
								x_ords = row['bbox'][::2]	#	Every other element starting from first - these are the X px coords
								y_ords = row['bbox'][1::2]	#	Every other element starting from second - these are the y px coords
								boundingBox = True
						except(KeyError):
								# print('except(KeyError)')
								x_ords = np.array(row['relative_coordinates.segmentation'][::2], dtype=np.float32)	
								y_ords = np.array(row['relative_coordinates.segmentation'][1::2], dtype=np.float32) 
								boundingBox = False

						pad_ones = np.ones(y_ords.shape[0])	#	padding so we get a 3 x num xy-paris matrix for our matmul
						# print('pad_ones done')
						xy_mat = np.vstack([x_ords, y_ords, pad_ones])	#	Add a row of ones for matmul
						# print('xy_mat done')
						geo_pairs = np.matmul(affine_mat, xy_mat)
						# print('geo_pairs done')
						#	matmul: https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html#coordinate-transformation

						print(boundingBox, 'boundingBox')
						if boundingBox:
								minx = np.min(geo_pairs[0,:]) # lowest value in the row of x values
								maxx = np.max(geo_pairs[0,:])
								miny = np.min(geo_pairs[1,:])
								maxy = np.max(geo_pairs[1,:])			  
								#	now form up the pairs for polygons - preferred order anti-clockwise from lowerleft
								poly_points = zip([minx, minx, maxx, maxx, minx],
												  [miny, maxy, maxy, miny, miny])
								return Polygon(poly_points)
						elif not boundingBox:
								geo_x = geo_pairs[0,:]
								geo_y = geo_pairs[1,:]

								# now form up the pairs for polygons - preferred order anti-clockwise from lowerleft
								poly_points = [Point(xy) for xy in zip(geo_x.tolist(), geo_y.tolist())]
								if len(poly_points) >= 3:
										s = Polygon(poly_points)
										'''
										simplify: reduce the number of nodes 
										all points in the simplified object will be within the
										tolerance distance of the original polygon
										'''
										ss = s.simplify(tolerance=0.05)
										# print(ss)
										return ss
								else:
										time.sleep(10)
										print('calc_geoobj_affine')
										print('No polygons',len(poly_points))
										return None							
				else:
						return None

		def to_coco_bbox(row):
				'''
				Return a COCO box
				darknet box format: normalized (x_ctr, y_ctr, w, h)
				coco box format: unnormalized (xmin, ymin, xmax, ymax)
				'''
				img_width = row.width
				img_height = row.height
				relative_h = row['relative_coordinates.height']
				relative_w = row['relative_coordinates.width']
				relative_x = row['relative_coordinates.center_x']
				relative_y = row['relative_coordinates.center_y']

				box_width = round(relative_w * img_width, 2)
				box_height = round(relative_h * img_height, 2)
				box_x_ctr = relative_x * img_width
				box_y_ctr = relative_y * img_height
				xmin = round(box_x_ctr - box_width / 2., 2)
				xmax = round(box_x_ctr + box_width / 2., 2)
				ymin = round(box_y_ctr - box_height / 2., 2)
				ymax = round(box_y_ctr + box_height / 2., 2)
				# We could return the xmin, ymin, xmax, ymax pairs and apply our cal_goebox_affine
				bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)    
				return bbox

		def affine_to_coco_bbox(row, ext=PREDICTIONS_IMG_WORLDFILE):
				'''
				Convert the affine matrix to COCO box format			
				darknet box format: normalized (x_ctr, y_ctr, w, h)
				coco box format: unnormalized (xmin, ymin, xmax, ymax)				
	
				Return a COCO box 
				'''
				img_width = row.width
				img_height = row.height

				#	get the worldfile
				world_file = row["filename"][:-3] + ext
				# wm = np.loadtxt(world_file, delimiter = '\n')
				wm = np.genfromtxt(world_file, delimiter = '\n')
				'''
				lines:params 1:a, 2:d, 3:b, 4:e, 5:c, 6:f
				1: x resolution, 				  2: amount of translation, 
				3: amount of rotation, 			  4: negative of the y resolution, 
				5: xMin coordinate( upper left ), 6: yMax coordinate (upper left)
				'''
				afmat = wm[0], wm[2], wm[4], wm[1], wm[3], wm[5] 

				minx = wm[4] - ( wm[0] / 2 ) 
				maxy = wm[5] - ( wm[3] / 2 )

				maxx = ( wm[4] + ( img_width * wm[0] ) ) - ( wm[0] / 2 )
				miny = ( wm[5] + ( img_height * wm[3] ) ) - ( wm[3] / 2 )

				bbox = np.array([minx, maxy, maxx, miny], dtype=np.float32)

				poly_points = zip([minx, minx, maxx, maxx, minx],
								  [miny, maxy, maxy, miny, miny])

				#	now form up the pairs for polygons - preferred order anti-clockwise from lowerleft
				return Polygon(poly_points)

		#	check that a folder to store prediction geojsons exist
		if not os.path.exists(PRED_FOLDER):
				os.makedirs(PRED_FOLDER)	

		#	open worldfile and get relevant data
		results_df['affine_elem'] = results_df.parallel_apply(get_affine_txt, axis = 1)
		# print(results_df['affine_elem'])
		results_df.iloc[0].affine_elem
		results_df_seg = deepcopy(results_df)
		results_df_affine = deepcopy(results_df)

		#	BOUNDING BOX CONVERSION/GEOREFERENCING ##########################################################
		#	for each image, convert each darknet format box to coco box
		results_df['bbox'] = results_df.parallel_apply(to_coco_bbox, axis = 1)
		print('bbox done')
		#	calculate the geo box from the coco box and the worldfile
		results_df['geo_coordinates'] = results_df.parallel_apply(MinMax, axis = 1)
		print('geo-referenced done')		
		results_df['geometry'] = results_df.parallel_apply(calc_geoobj_affine, axis = 1)

		#	SAVE ALL THE PREDICTION BOUNDING BOXES 
		results_df.columns		
		# Clean the column types for export
		out_df = results_df[['class_id', 'name', 'confidence', 'filename', 'geometry']].copy()
		out_df['class_id'] = out_df['class_id'].astype('int')
		out_df['name'] = out_df['name'].astype('str')
		out_df['confidence'] = out_df['confidence'].astype('float')
		out_df['filename'] = out_df['filename'].astype('str')
		print('Saving non-filtered data frame to GeoJSON',datetime.now())
		out_df.dtypes
		# Add the option to change this based on the region. E.g. Sinarmas.
		gdf = gpd.GeoDataFrame(out_df, crs='EPSG:2193')
		gdf = gdf.to_crs(2193)

		if CUSTOM_TRAINER:
				gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_box_custom.geojson' ,driver='GeoJSON')
		else: 
				gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_box_default.geojson' ,driver='GeoJSON')
		print('SAVED non-filtered data to GeoJSON', datetime.now())

		# print(results_df)
		# print(out_df)


		if annotation_type == 'instance':
				#	SEGMENTATION POLYGON CONVERSION/GEOREFERENCING ##################################################
				#	for each image, get the coco polygon
				results_df_seg['poly'] = results_df_seg.parallel_apply(to_coco_bbox, axis = 1)
				print('bbox done')
				#	calculate the geo polygon from the coco polygon and the worldfile
				results_df_seg['geo_coordinates'] = results_df_seg.parallel_apply(MinMax, axis = 1)
				print('geo-referenced done')
				results_df_seg['geometry'] = results_df_seg.parallel_apply(calc_geoobj_affine, axis = 1)
				print('geometry done')

				#	SAVE ALL THE PREDICTION SEGMENTATIONs
				results_df_seg.columns
				# Clean the column types for export
				out_df = results_df_seg[['class_id', 'name', 'confidence', 'filename', 'geometry']].copy()
				out_df['class_id'] = out_df['class_id'].astype('int')
				out_df['name'] = out_df['name'].astype('str')
				out_df['confidence'] = out_df['confidence'].astype('float')
				out_df['filename'] = out_df['filename'].astype('str')
				# print(out_df['geometry'])
				print('Saving non-filtered data frame to GeoJSON',datetime.now())
				out_df.dtypes
				gdf = gpd.GeoDataFrame(out_df, crs='EPSG:2193')
				gdf = gdf.to_crs(2193)

				if CUSTOM_TRAINER:
						gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_seg_custom.geojson' ,driver='GeoJSON')
				else: 
						gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_seg_default.geojson' ,driver='GeoJSON')
				print('SAVED non-filtered data to GeoJSON', datetime.now())

		#	AFFINE BOX CONVERSION/GEOREFERENCING ############################################################
		#	for each image convert the worldfile information to coco box box 
		results_df_affine['geometry'] = results_df_affine.parallel_apply(affine_to_coco_bbox, axis=1)
		print('affine bbox done', results_df_affine['geometry'])

		#	SAVE ALL THE PREDICTION AFFINE BOUNDING BOXES
		results_df_seg.columns
		# Clean the column types for export
		out_df = results_df_affine[['class_id', 'name', 'confidence', 'filename', 'geometry']].copy()
		out_df['class_id'] = out_df['class_id'].astype('int')
		out_df['name'] = out_df['name'].astype('str')
		out_df['confidence'] = out_df['confidence'].astype('float')
		out_df['filename'] = out_df['filename'].astype('str')
		# print(out_df['geometry'])
		print('Saving non-filtered data frame to GeoJSON',datetime.now())
		out_df.dtypes
		gdf = gpd.GeoDataFrame(out_df, crs='EPSG:2193')
		gdf = gdf.to_crs(2193)		

		if CUSTOM_TRAINER:
				gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_affine_custom.geojson' ,driver='GeoJSON')
		else: 
				gdf.to_file(f'{PRED_FOLDER}/{PREDICTION_PROPERTY}_affine_default.geojson' ,driver='GeoJSON')
		print('SAVED non-filtered data to GeoJSON', datetime.now())

class Detector:
	def __init__(self, model_type='box'):
		pretrained_model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
		self.cfg = get_cfg()
		self.cfg.merge_from_file("tree_detection_small.yaml")
		self.cfg.MODEL.WEIGHTS = "default_model_final.path"
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
		self.cfg.MODEL.DEVICE = "cpu"
		self.predictor = DefaultPredictor(self.cfg)

	def onimage(self, imagePath):
		image = cv2.imread(imagePath)
		predictions = self.predictor(image)
		viz = Visualizer(image[:,:,::-1],metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),scale=1.2)
		output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
		filename = 'result.jpg'
		cv2.imwrite(filename,output.get_image()[:,:,::-1])
		cv2.waitkey(0)
		cv2.destroyAllWindows()