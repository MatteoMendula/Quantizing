
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time
import torch
from io import BytesIO # "import StringIO" directly in python2
from PIL import Image
import os

def get_compressed_image(image_path, quality=40):
    from io import BytesIO # "import StringIO" directly in python2
    from PIL import Image
    import os
    im1 = Image.open(image_path)

    # here, we create an empty string buffer    
    buffer = BytesIO()
    im1.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    image = Image.open(buffer)
    return image

def get_coco_object_dictionary():
    import os
    file_with_coco_names = "category_names.txt"

    if not os.path.exists(file_with_coco_names):
        print("Downloading COCO annotations.")
        import urllib
        import zipfile
        import json
        import shutil
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "cocoanno.zip")
        with zipfile.ZipFile("cocoanno.zip", "r") as f:
            f.extractall()
        print("Downloading finished.")
        with open("annotations/instances_val2017.json", 'r') as COCO:
            js = json.loads(COCO.read())
        class_names = [category['name'] for category in js['categories']]
        open("category_names.txt", 'w').writelines([c+"\n" for c in class_names])
        os.remove("cocoanno.zip")
        shutil.rmtree("annotations")
    else:
        class_names = open("category_names.txt").readlines()
        class_names = [c.strip() for c in class_names]
    return class_names

def plot_results(best_results, inputs, classes_to_labels):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    print(inputs.squeeze(0).shape)
    ax.imshow(torch.transpose(inputs.squeeze(0), 0, 2).transpose(0, 1))
    # ...with detections
    bboxes = best_results[0]["boxes"].cpu().detach().numpy().tolist()
    classes = best_results[0]["labels"].cpu().detach().numpy().tolist()
    confidences = best_results[0]["scores"].cpu().detach().numpy().tolist()
    for idx in range(len(bboxes)):
        if confidences[idx] < 0.7:
            continue

        if classes[idx] > len(classes_to_labels):
            continue

        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


img_path = "../images/kitti_1.png"
jpeg_compression = 20
img = read_image(img_path)

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.cuda()
model.eval()

print(model)

# jpeg_compression image
img = get_compressed_image(img_path, jpeg_compression)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img).cuda()]

# Step 4: Use the model and visualize the prediction
start_time = time.time()
prediction = model(batch)
print("--- %s seconds ---" % (time.time() - start_time))
print(prediction)

classes_to_labels= get_coco_object_dictionary()
plot_results(prediction, batch[0].cpu(), classes_to_labels)



