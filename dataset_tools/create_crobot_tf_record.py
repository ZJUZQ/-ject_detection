# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert the crobot dataset to TFRecord for object_detection.

Example usage:
	python object_detection/dataset_tools/create_crobot_tf_record.py \
    --label_map_path=object_detection/data/crobot_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
"""

import hashlib
import io
import logging
import os
import random
import re

import json
import numpy as np
import PIL.Image
import tensorflow as tf

import _init_path
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to crobot dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
										'Path to label map proto')
flags.DEFINE_boolean('bbox_only', True, 'If True, generates bounding boxes only '
										 'Otherwise generates bounding boxes (as '
										 'well as segmentations for crobot bodies).	Note that '
										 'in the latter case, the resulting files are much larger.')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
										'segmentation masks. Options are "png" or "numerical".')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
					   label_map_dict,
					   image_subdirectory,
					   example,
					   ignore_difficult_instances=False,
					   bbox_only=True,
					   mask_type='png'):
	"""Convert XML derived dict to tf.Example proto.

	Notice that this function normalizes the bounding box coordinates provided
	by the raw data.

	Args:
		data: dict holding PASCAL XML fields for a single image (obtained by
			running dataset_util.recursive_parse_xml_to_dict)
		label_map_dict: A map from string label names to integers ids.
		image_subdirectory: String specifying subdirectory within the
			Pascal dataset directory holding the actual image data.
		ignore_difficult_instances: Whether to skip difficult instances in the
			dataset	(default: False).
		faces_only: If True, generates bounding boxes for pet faces.	Otherwise
			generates bounding boxes (as well as segmentations for full pet bodies).
		mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
			smaller file sizes.

	Returns:
		example: The converted tf.Example.

	Raises:
		ValueError: if the image pointed to by data['filename'] is not a valid JPEG
	"""
	imgName = '{}.jpg'.format(example)
	img_path = os.path.join(image_subdirectory, imgName)
	with tf.gfile.GFile(img_path, 'rb') as fid:
		encoded_jpg = fid.read()
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = PIL.Image.open(encoded_jpg_io)
	if image.format != 'JPEG':
		raise ValueError('Image format not JPEG')
	key = hashlib.sha256(encoded_jpg).hexdigest()

	width = int(data['width'])
	height = int(data['height'])

	xmins = []
	ymins = []
	xmaxs = []
	ymaxs = []
	classes = []
	classes_text = []
	masks = [] # if image has several objects, each obj one mask
	if 'shapes' in data:
		for shape in data['shapes']:

			xmin = float(shape['bbox']['xmin'])
			xmax = float(shape['bbox']['xmax'])
			ymin = float(shape['bbox']['ymin'])
			ymax = float(shape['bbox']['ymax'])

			xmins.append(xmin / width)
			ymins.append(ymin / height)
			xmaxs.append(xmax / width)
			ymaxs.append(ymax / height)

			class_name = shape['label']
			classes_text.append(class_name.encode('utf8'))
			classes.append(label_map_dict[class_name])

			if not bbox_only:
				mask = np.zeros((height, width), dtype=np.uint8)
				mask = PIL.Image.fromarray(mask)
				xy = list(map(tuple, shape['points']))
				ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
				mask_np = np.asarray(mask)

				# 0 is background
				mask_remapped = (mask_np != 0).astype(np.uint8)
				masks.append(mask_remapped)

	feature_dict = {
			'image/height': dataset_util.int64_feature(height),
			'image/width': dataset_util.int64_feature(width),
			'image/filename': dataset_util.bytes_feature(
					imgName.encode('utf8')),
			'image/source_id': dataset_util.bytes_feature(
					imgName.encode('utf8')),
			'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
			'image/encoded': dataset_util.bytes_feature(encoded_jpg),
			'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
			'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
			'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
			'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
			'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
			'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
			'image/object/class/label': dataset_util.int64_list_feature(classes),
	}
	if not bbox_only:
		if mask_type == 'numerical':
			mask_stack = np.stack(masks).astype(np.float32)
			masks_flattened = np.reshape(mask_stack, [-1])
			feature_dict['image/object/mask'] = (
					dataset_util.float_list_feature(masks_flattened.tolist()))
		elif mask_type == 'png':
			encoded_mask_png_list = []
			for mask in masks:
				img = PIL.Image.fromarray(mask)
				output = io.BytesIO()
				img.save(output, format='PNG')
				encoded_mask_png_list.append(output.getvalue())
			feature_dict['image/object/mask'] = (
					dataset_util.bytes_list_feature(encoded_mask_png_list))

	example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
	return example


def create_tf_record(output_filename,
					 label_map_dict,
					 annotations_dir,
					 image_dir,
					 examples,
					 bbox_only=True,
					 mask_type='png'):
	"""Creates a TFRecord file from examples.

	Args:
		output_filename: Path to where output file is saved.
		label_map_dict: The label map dictionary.
		annotations_dir: Directory where annotation files are stored.
		image_dir: Directory where image files are stored.
		examples: Examples to parse and save to tf record.
		bbox_only: If True, generates bounding boxes for crobot. Otherwise
			generates bounding boxes (as well as segmentations).
		mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
			smaller file sizes.
	"""
	writer = tf.python_io.TFRecordWriter(path=output_filename)
	for idx, example in enumerate(examples):
		print('{} / {}'.format(idx, len(examples)), end='\r', flush=True)
		if idx % 100 == 0:
			logging.info('On image %d of %d', idx, len(examples))

		json_path = os.path.join(annotations_dir, example + '.json')

		if not os.path.exists(json_path):
			logging.warning('Could not find %s, ignoring example.', json_path)
			continue

		with open(json_path, 'r') as f:
			data = json.load(f)

		try:
			tf_example = dict_to_tf_example(
					data,
					label_map_dict,
					image_dir,
					example,
					bbox_only=bbox_only,
					mask_type=mask_type)
			writer.write(tf_example.SerializeToString())
		except ValueError:
			logging.warning('Invalid example: %s, ignoring.', xml_path)
	print('')

	writer.close()


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
	data_dir = FLAGS.data_dir
	label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

	logging.info('Reading from crobot dataset.')
	image_dir = os.path.join(data_dir, 'images')
	annotations_dir = os.path.join(data_dir, 'annotations')

	examples_path = os.path.join(data_dir, 'trainval.txt')
	examples_list = dataset_util.read_examples_list(examples_path)

	# Test images are not included in the downloaded data set, so we shall perform
	# our own split.
	random.seed(42)
	random.shuffle(examples_list)
	num_examples = len(examples_list)
	num_train = int(0.7 * num_examples)
	train_examples = examples_list[:num_train]
	val_examples = examples_list[num_train:]
	logging.info('%d training and %d validation examples.',
							 len(train_examples), len(val_examples))

	train_output_path = os.path.join(FLAGS.output_dir, 'crobot_train.record')
	val_output_path = os.path.join(FLAGS.output_dir, 'crobot_val.record')
	if not FLAGS.bbox_only:
		train_output_path = os.path.join(FLAGS.output_dir, 'crobot_train_with_masks.record')
		val_output_path = os.path.join(FLAGS.output_dir, 'crobot_val_with_masks.record')

	print('Create train set TFRecord')
	create_tf_record(
			train_output_path,
			label_map_dict,
			annotations_dir,
			image_dir,
			train_examples,
			bbox_only=FLAGS.bbox_only,
			mask_type=FLAGS.mask_type)
	print('Create val set TFRecord')
	create_tf_record(
			val_output_path,
			label_map_dict,
			annotations_dir,
			image_dir,
			val_examples,
			bbox_only=FLAGS.bbox_only,
			mask_type=FLAGS.mask_type)


if __name__ == '__main__':
	tf.app.run()