import os
import sys
os.system("conda create -n tf python=3.7")
os.system("conda activate tf")
os.system("pip install pandas")
os.system("pip install scipy")
os.system("pip install tensorflow==1.15.0")
os.system("pip install numpy==1.17.5")
os.system("pip install tf_slim")
os.system("pip install protobuf==3.20.1")

# For the fruit model included in the repo below we have 240 training images
# For faster training time, images should be resized to 300x300 and then annotated
# Images should contain the objects of interest at various scales, angles, lighting conditions, locations
# For acceptable results - mAP@0.5 of 0.9 the model was trained with batch size of 24
# and 5000 steps. this takes about 1h using 2 augmentations. 
# using 5 augmentations it takes about 2h 
num_steps = 5000  # A step means using a single batch of data. larger batch, less steps required
#Number of evaluation steps.
num_eval_steps = 50
#Batch size 24 is a setting that generally works well. can be changed higher or lower 
MODELS_CONFIG = {
        'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 24
    }
}
selected_model = 'ssd_mobilenet_v2'

# Name of the object detection model to use.
MODEL = MODELS_CONFIG[selected_model]['model_name']

# Name of the pipline file in tensorflow object detection API.
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

# Training batch size fits in Colab's GPU memory for selected model.
batch_size = MODELS_CONFIG[selected_model]['batch_size']

## Clone the object_detection_demo_flow repository

if not (os.path.isdir("./content")): os.makedirs("./content") 
os.chdir("./content")
# os.system("git clone https://github.com/GotG/object_detection_demo_flow")
# os.chdir("object_detection_demo_flow")
# os.system("git pull")

#To train on your own data:
#Remove repo data (images) for training/testing/final testing
# os.system("rm -r ./object_detection_demo_flow/data/images/final_test/")
# os.system("rm -r ./object_detection_demo_flow/data/images/train/" )
# os.system("rm -r ./object_detection_demo_flow/data/images/test/")

# """## Install Tensorflow Object Detection API"""

os.system("git clone https://github.com/tensorflow/models.git")
os.chdir("./models")
os.system("git checkout 58d19c67e1d30d905dd5c6e5092348658fed80af")
os.system("sudo apt-get install -qq protobuf-compiler python-pil python-lxml python-tk")
os.system("pip install -q Cython contextlib2 pillow lxml matplotlib")
os.system("pip install -q pycocotools")
os.chdir("./research")
os.system("protoc object_detection/protos/*.proto --python_out=.")
# sys.path.insert(0,os.getcwd())
# sys.path.insert(0,os.getcwd()+"/slim")
# sys.path.insert(0,os.getcwd()+"/object_detection")
# os.environ['PYTHONPATH'] += ':/content/models/research:/content/models/research/slim'
# print("current path is: "+os.getcwd())
os.chdir("../../..")
# print("current path is: "+os.getcwd())
# os.environ['PYTHONPATH'] += ':'+os.getcwd()
# print(sys.path)
# print("things go bananas after this line")
os.system("python content/models/research/object_detection/builders/model_builder_test.py")



# # Convert train folder annotation xml files to a single csv file,
# # generate the `label_map.pbtxt` file to `data/` directory as well.


os.system("python xml_to_csv.py -i images/train -o images/annotations/train_labels.csv -l images/annotations")
# 
# # Convert test folder annotation xml files to a single csv.
os.system("python xml_to_csv.py -i images/test -o images/annotations/test_labels.csv")
# 
# # Generate `train.record`
os.system("python generate_tfrecord.py --csv_input=images/annotations/train_labels.csv --output_path=images/annotations/train.record --img_path=images/train --label_map images/annotations/label_map.pbtxt")
# 
# # Generate `test.record`
os.system("python generate_tfrecord.py --csv_input=images/annotations/test_labels.csv --output_path=images/annotations/test.record --img_path=images/test --label_map images/annotations/label_map.pbtxt")
# 
# # Set the paths
test_record_fname = './images/annotations/test.record'
train_record_fname = './images/annotations/train.record'
label_map_pbtxt_fname = './images/annotations/label_map.pbtxt'

"""## Download the Mobilenet SSD v2 Model"""

# Commented out IPython magic to ensure Python compatibility.

import shutil
import glob
import urllib.request
import tarfile
MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = './content/models/research/pretrained_model'

if not (os.path.exists(MODEL_FILE)):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar = tarfile.open(MODEL_FILE)
tar.extractall()
tar.close()

os.remove(MODEL_FILE)
if (os.path.exists(DEST_DIR)):
    shutil.rmtree(DEST_DIR)

os.rename(MODEL, DEST_DIR)
cmd = "echo " + DEST_DIR
cmd2 = "ls -alh " + DEST_DIR
os.system(cmd)
os.system(cmd2)

#TF pretrained model checkpoint
fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
fine_tune_checkpoint

pipeline_fname = os.path.join('./content/models/research/object_detection/samples/configs/', pipeline_file)


assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

import re
iou_threshold = 0.50
num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    # Set number of classes num_classes.
    s = re.sub('iou_threshold: [0-9].[0-9]+',
               'iou_threshold: {}'.format(iou_threshold), s)
    
    f.write(s)

#Have a look at the config file with various settings
os.system("cat" + pipeline_fname)

model_dir = 'training/'
# Optionally remove content in output model directory for a fresh start.
os.system("rm -rf "+model_dir)
os.makedirs(model_dir, exist_ok=True)
os.system("python ./content/models/research/object_detection/model_main.py --pipeline_config_path=" + pipeline_fname +"\
    --model_dir="+model_dir +"\
    --alsologtostderr \
    --num_train_steps=5000 \
    --num_eval_steps=50")

#model dir check for the trained model
os.system("ls"+ model_dir)

"""## Export a Trained Inference Graph"""

# Commented out IPython magic to ensure Python compatibility.
import re
import numpy as np
import os
 
output_directory = './fine_tuned_model'
# output_directory = '/content/gdrive/My\ Drive/data/'

model_dir = 'training/'
 
lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

MODELS_CONFIG = {
        'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 24
    }
}
selected_model = 'ssd_mobilenet_v2'

# Name of the pipline file in tensorflow object detection API.
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

pipeline_fname = os.path.join('./content/models/research/object_detection/samples/configs/', pipeline_file)
 
last_model_path = os.path.join(model_dir, last_model)
print(last_model_path)
os.system("python ./content/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path="+pipeline_fname +" \
    --output_directory="+output_directory +" \
    --trained_checkpoint_prefix="+last_model_path +"")

import os
pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph.pb")
assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)
# !ls -alh {pb_fname}

"""## Running Inference"""

import os
import glob

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = pb_fname

label_map_pbtxt_fname = './images/annotations/label_map.pbtxt'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = label_map_pbtxt_fname

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR =  './images/final_test'

assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
print(TEST_IMAGE_PATHS)

# Commented out IPython magic to ensure Python compatibility.
os.chdir("./content/models/research/object_detection")

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# This is needed to display the images.
# %matplotlib inline
os.system("pip install matplotlib==2.2.2")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

os.chdir("../../../..")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (24, 20)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
 


for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    print(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # print(output_dict)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=.01,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()