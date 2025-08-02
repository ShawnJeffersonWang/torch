import torch
import cv2
import numpy as np
import torchvision
from model.yolo import YOLO

def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]

    return outputs

def load_image(img_path, input_size):
    img = cv2.imread(img_path)  # BGR
    assert img is not None, f"Image Not Found {img_path}"

    h0, w0 = img.shape[:2]  # original height and width
    r = input_size / max(h0, w0)  # resize image to input_size
    if r != 1:  # resize
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = img.shape[:2]  # new h,w
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float()
    img /= 255.0  # normalize to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # add batch dimension

    return img, (h0, w0), imgsz  # return original size and resized shape

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def detect_single_image(model, img_path, input_size):
    img, original_shape, resized_shape = load_image(img_path, input_size)

    img = img.half().cuda()  # to float16
    model.half().cuda().eval()

    with torch.no_grad():
        outputs = model(img)

    outputs = non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.45)

    # outputs is a list (batch size=1),
    # each element is detections: (num_boxes, 6) [x1, y1, x2, y2, conf, class]
    detections = outputs[0]

    print("detections.shape:", detections.shape)

    if detections is None or len(detections) == 0:
        print("No objects detected.")
        return

    # Scale boxes back to original image size
    detections[:, :4] = scale_coords(resized_shape, detections[:, :4], original_shape).round()

    print("Detections:")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        print(f"Class: {int(cls)}, Confidence: {conf:.2f}, BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    # 加载原图
    img0 = cv2.imread(img_path)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{int(cls)} {conf:.2f}"

        # 画矩形框
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽2
        # 写类别+置信度
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)  # 蓝色字体

    # 保存检测结果
    cv2.imwrite('bus_detected.jpg', img0)
    print("Detection results saved to bus_detected.jpg")

if __name__ == "__main__":
    model_path = './yolo_v8_n.pth'
    image_path = './bus.jpg'
    input_size = 640  # same as --input-size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 实例化自己的YOLO模型
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    num_classes = 80

    model = YOLO(width, depth, num_classes).to(device).float()

    # 加载保存的模型权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    detect_single_image(model, image_path, input_size)
