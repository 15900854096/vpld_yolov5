# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import copy
import yaml 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, elimination_holes, non_max_suppression_arr, non_max_suppression_arr_new)
from utils.debug import *
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

with open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore') as f:
    hyp = yaml.safe_load(f)
    
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # fo = open(r"./onnx_output/input.txt", "w")
            # im1=np.array(im.cpu())
            # print(im1.shape)
            # for i in range(im1.shape[0]):
            #         for j in range(im1.shape[1]):
            #             for k in range(im1.shape[2]):
            #                 for l in range(im1.shape[3]):
            #                     fo.write("%0.4f"%im1[i,j,k,l])
            #                     fo.write("\n")
            pred = model(im, augment=augment, visualize=visualize)
            # pred1=np.array(pred[0].cpu())
            # print(pred1.shape)
            # fo = open(r"./onnx_output/output.txt", "w")
            # for i in range(pred1.shape[0]):
            #         for j in range(pred1.shape[1]):
            #             for k in range(pred1.shape[2]):
            #                     fo.write("%0.4f"%pred1[i,j,k])
            #                     fo.write("\n")
            #sys.exit()
        # NMS
        with dt[2]: #pred[0]["vpld"] type is list
            pred[0]["vpld"] = non_max_suppression(pred[0]["vpld"], conf_thres, iou_thres, imgsz[0], classes, agnostic_nms, max_det=max_det)
            if(hyp["task_fs"]):
                pred[0]["fs"][0] = torch.max(pred[0]["fs"][0],dim=1).indices.cpu()  #1CHW float--->1HW index， max不仅会求出dim维度的那个最大值，而且还会消除dim这个维度          
            if(hyp["task_ss"]):
                #pred[0]["ss"][0] = non_max_suppression_arr(pred[0]["ss"], conf_thres, iou_thres, imgsz[0], classes, agnostic_nms, max_det=max_det) #1CHW float--->1HW index， max不仅会求出dim维度的那个最大值，而且还会消除dim这个维度          
                pred[0]["ss"][0] = non_max_suppression_arr_new(pred[0]["ss"], hyp["ss_arrow_label"], conf_thres, iou_thres, imgsz[0], classes, agnostic_nms, max_det=max_det) #1CHW float--->1HW index， max不仅会求出dim维度的那个最大值，而且还会消除dim这个维度          


        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        for i, det in enumerate(pred[0]["vpld"]):  # per image, 因为等同于bachsize=1,所以i永远是0
            # 0 1  2   3  4  5   6   7   8   9   10
            # x y len c1 s1 ADc ADs BCc BCs conf cls x&y:base_640  others:normal 1
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if (hyp["task_fs"]):
                mask = pred[0]["fs"][0][i][:, :, np.newaxis]#hw->hw1  为了后面resize
                mask = np.array(mask).astype("float")
                pre_color = copy.deepcopy(im0)
                mask = cv2.resize(mask, (im0.shape[0], im0.shape[1]))
                mask = mask.astype(np.uint8)
                mask = elimination_holes(mask, hyp["fs_num_class"], hyp["holes_rate"])
                
                bchanel =  np.vectorize(dict_col_b.get)(mask)
                gchanel =  np.vectorize(dict_col_g.get)(mask)
                rchanel =  np.vectorize(dict_col_r.get)(mask)
                pre_color[:,:,0], pre_color[:,:,1], pre_color[:,:,2]  = bchanel, gchanel, rchanel
                
                im0 = cv2.addWeighted(pre_color,0.3,im0,0.7,0)
                
            if (hyp["task_ss"]):
                arrs = np.array(pred[0]["ss"][0][i].cpu())
                color = (0,255,0)
                if(len(arrs)):
                    arrs[:,0:6] *= np.array([im0.shape[0]/imgsz[0], im0.shape[1]/imgsz[0], im0.shape[0]/imgsz[0], im0.shape[1]/imgsz[0], im0.shape[0]/imgsz[0], im0.shape[1]/imgsz[0]])
                    for Ax, Ay, Bx, By, Cx, Cy, conf, cls in reversed(arrs):
                        Ax, Ay, Bx, By, Cx, Cy = round(Ax), round(Ay), round(Bx), round(By), round(Cx), round(Cy)
                        cv2.line(im0, (Ax, Ay), (Bx, By), color, 2, 4)
                        if(cls==hyp["ss_arrow_label"]):
                            cv2.line(im0, (Bx, By), (Cx, Cy), color, 2, 4)
                            cv2.line(im0, (Cx, Cy), (Ax, Ay), color, 2, 4)
                    
                
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            #annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :2] = scale_boxes(im.shape[2:], det[:, :2], im0.shape).round() #base_640 to base_600

                # Print results
                for c in det[:, 10].unique():
                    n = (det[:, 10] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xylenc1s1ADcADsBCcBCs, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    #    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #    with open(f'{txt_path}.txt', 'a') as f:
                    #        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        x,y,leng,c1,s1,ADc,ADs,BCc,BCs = xylenc1s1ADcADsBCcBCs
                        h,w,c = im0.shape
                        leng = leng*w
                        p1 = (round(float(x)) , round(float(y)))
                        p2 = (round(float(x+leng*c1)) , round(float(y+leng*s1)))
                        p3 = (round(float(p2[0]+100*BCc)) , round(float(p2[1]+100*BCs)))
                        p4 = (round(float(p1[0]+100*ADc)) , round(float(p1[1]+100*ADs)))
                        #print(p1,p2,p3,p4)
                        if(cls==0):
                            color = (0,255,0)
                        else:
                            color = (0,0,255)    
                        cv2.arrowedLine(im0, p1, p2, color, 2, 4)
                        cv2.arrowedLine(im0, p2, p3, color, 2, 4)
                        cv2.arrowedLine(im0, p1, p4, color, 2, 4)
                        cv2.putText(im0,"%0.2f"%float(conf),(int(0.5*(p1[0]+p2[0])),int(0.5*(p1[1]+p2[1]))),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
                        #c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                    #if save_crop:
                    #    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            #im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
