import os
import sys
import torch
from subprocess import call
from PIL import Image, ImageDraw


def print_cuda_statistics():
    print('------Python------Version------ {}'.format(sys.version))
    print('------PyTorch-----Version------ {}'.format(torch.__version__))
    print('------CUDA--------Version------')
    call(["cat", "/usr/local/cuda/version.txt"])
    print('------CUDNN-------Version------ {}'.format(torch.backends.cudnn.version()))
    print('------CUDA-Devices-Number------ {}'.format(torch.cuda.device_count()))
    print('-----------Devices-------------')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('------Available---Devices------ {}'.format(torch.cuda.device_count()))
    print('------Current-CUDA-Device------ {}'.format(torch.cuda.current_device()))


def create_dirs(*dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


class ImageProcessor(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def normalize(self, image):
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.float().div(255).view(image.size[1], image.size[0], 3)
        data = data.permute(2, 0, 1)

        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        return data

    def show_detections(self, image, detections, path):
        """
        Show detection results on an image.
        :param image: PIL image object
        :param detections: list of detection result dictionaries.
        """
        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1]), '[{}]'.format(detection['class']),
                      fill=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1] + 10), '{:.2}'.format(detection['score']),
                      fill=(255, 255, 255, alpha))
        image = Image.alpha_composite(image, overlay)
        image.save(os.path.join(path, 'test.png'))