# Person_Follower_Robot

## Models to download: 
- Google Drive Link: https://drive.google.com/drive/folders/17MSviLBLsBMN5Wo9jXc0iX2BPkxaZm_5?usp=drive_link

## Dataset to download:
- Dataset Link: https://app.roboflow.com/ds/xjijjOCjU4?key=LaxGN0HYTQ
  
## Features
- Train a MobileNet-SSD model to detect persons from a custom dataset.
- Annotate dataset using Roboflow (VOC export).
- Train the SSD model using the PyTorch-SSD repository.
- Convert trained .pth weights into ONNX format.
- Deploy ONNX model on Raspberry Pi for real-time detection.
- Robot movement logic:
  - Center = Move Forward
  - Left = Turn Left
  - Right = Turn Right
  - No Detection = Stop
- Tested using ONNX Runtime inference.

## Requirements
- Python 3.7+
- PyTorch + torchvision
- OpenCV
- ONNX Runtime
- Raspberry Pi (for deployment)
- Camera module
- Roboflow workspace
- Pretrained MobileNet-SSD base model
- VS Code or Google Colab for training

## Cloning the Original Repository
git clone https://github.com/qfgaohao/pytorch-ssd.git
cd pytorch-ssd

## Dataset
Collected:
- ~353 annotated images
- Labeled using Roboflow
- Exported in VOC XML format
- Automatically generated Train / Val / Test splits

Classes:
- person
(Note: VOC contains additional classes but they are unused)

Directory Structure:
dataset/VOC2007/
 ├── Annotations/
 ├── JPEGImages/
 ├── ImageSets/Main/train.txt
 ├── ImageSets/Main/val.txt
 ├── ImageSets/Main/test.txt
 ├── ImageSets/Main/trainval.txt

## Training the Model
Recommended Training Command:
python train_ssd.py --dataset_type voc --datasets dataset/VOC2007 \
--validation_dataset dataset/VOC2007 \
--net mb1-ssd \
--pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth \
--batch_size 4 \
--num_epochs 30 \
--scheduler cosine \
--lr 0.001 \
--num_workers 0 \
--freeze_base_net

Training Notes:
- Base MobileNet layers frozen (--freeze_base_net)
- Stable loss reached around epoch 25
- Best checkpoint file example:
  models/mb1-ssd-Epoch-25-Loss-4.4421718915303545.pth

## Converting the Model to ONNX
Conversion Command:
python convert_to_onnx.py models/mb1-ssd-Epoch-25-Loss-4.4421718915303545.pth \
models/voc-model-labels.txt \
--onnx models/person_follower.onnx

Output File:
models/person_follower.onnx

## Testing the ONNX Model
Command:
python test_onnx.py

The test script will:
- Load the ONNX model
- Preprocess the input image
- Run inference using ONNX Runtime
- Draw bounding boxes for detected objects

## Robot Following Logic
Behavior Table:
Person Position   | Robot Action
------------------|--------------
Center            | Forward
Left              | Turn Left
Right             | Turn Right
None              | Stop

Example Logic:
if not detected:
    stop()
elif x_center < left_threshold:
    turn_left()
elif x_center > right_threshold:
    turn_right()
else:
    move_forward()
    
## References
- PyTorch SSD repository: https://github.com/qfgaohao/pytorch-ssd
- MobileNet-SSD research paper
- ONNX Runtime documentation
- Roboflow tools and dataset pipeline
