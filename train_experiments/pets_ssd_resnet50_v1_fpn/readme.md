
## Getting the Oxford-IIIT Pets Dataset

In order to train a detector, we require a dataset of images, bounding boxes and
classifications. For this demo, we'll use the Oxford-IIIT Pets dataset. The raw
dataset for Oxford-IIIT Pets lives
[here](http://www.robots.ox.ac.uk/~vgg/data/pets/). You will need to download
both the image dataset [`images.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
and the groundtruth data [`annotations.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz)
to the `tensorflow/models/research/` directory and unzip them. This may take
some time.

``` bash
# From tensorflow/models/research/
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

## Generate TFRecord
The Tensorflow Object Detection API expects data to be in the TFRecord format,
so we'll now run the `create_pet_tf_record` script to convert from the raw
Oxford-IIIT Pet dataset into TFRecords. Run the following commands from the
`tensorflow/models/research/` directory:

``` bash
# From tensorflow/models/research/
python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

Note: It is normal to see some warnings when running this script. You may ignore
them.

Two TFRecord files named `pet_train.record` and `pet_val.record` should be
generated in the `tensorflow/models/research/` directory.


## Downloading a COCO-pretrained Model for Transfer Learning

Training a state of the art object detector from scratch can take days, even
when using multiple GPUs! In order to speed up training, we'll take an object
detector trained on a different dataset (COCO), and reuse some of it's
parameters to initialize our new model.

Download our [COCO-pretrained Faster R-CNN with Resnet-101
model](http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz).
Unzip the contents of the folder and copy the `model.ckpt*` files into your GCS
Bucket.

``` bash
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
```


## Configuring the Object Detection Pipeline

In the Tensorflow Object Detection API, the model parameters, training
parameters and eval parameters are all defined by a config file. More details
can be found [here](configuring_jobs.md). For this tutorial, we will use some
predefined templates provided with the source code. In the
`object_detection/samples/configs` folder, there are skeleton object_detection
configuration files. We will use `faster_rcnn_resnet101_pets.config` as a
starting point for configuring the pipeline. Open the file with your favourite
text editor.

We'll need to configure some paths in order for the template to work. Search the
file for instances of `PATH_TO_BE_CONFIGURED` and replace them with the
appropriate value (typically `gs://${YOUR_GCS_BUCKET}/data/`). Afterwards
upload your edited file onto GCS, making note of the path it was uploaded to
(we'll need it when starting the training/eval jobs).

``` bash
# From tensorflow/models/research/

# Edit the faster_rcnn_resnet101_pets.config template. Please note that there
# are multiple places where PATH_TO_BE_CONFIGURED needs to be set.
sed -i "s|PATH_TO_BE_CONFIGURED|"gs://${YOUR_GCS_BUCKET}"/data|g" \
    object_detection/samples/configs/faster_rcnn_resnet101_pets.config

```

## Starting training
```
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
```


## Exporting the Tensorflow Graph

After your model has been trained, you should export it to a Tensorflow
graph proto. First, you need to identify a candidate checkpoint to export. You
can search your bucket using the [Google Cloud Storage
Browser](https://console.cloud.google.com/storage/browser). The file should be
stored under `${YOUR_GCS_BUCKET}/train`. The checkpoint will typically consist of
three files:

* `model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001`
* `model.ckpt-${CHECKPOINT_NUMBER}.index`
* `model.ckpt-${CHECKPOINT_NUMBER}.meta`

After you've identified a candidate checkpoint to export, run the following
command from `tensorflow/models/research/`:

``` bash
# From tensorflow/models/research/
gsutil cp gs://${YOUR_GCS_BUCKET}/train/model.ckpt-${CHECKPOINT_NUMBER}.* .
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path object_detection/samples/configs/faster_rcnn_resnet101_pets.config \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_graphs
```

Afterwards, you should see a directory named `exported_graphs` containing the
SavedModel and frozen graph.