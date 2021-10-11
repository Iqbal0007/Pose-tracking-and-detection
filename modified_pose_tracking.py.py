import os
import warnings
from argparse import ArgumentParser

import cv2

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def merge(img1,img2,img3,img4,i):
    img1 = Image.open('G:/merging photos/videos/1/'+str(i)+'.jpg')
    img2 = Image.open('G:/merging photos/videos/2/'+str(i)+'.jpg')
    img3 = Image.open('G:/merging photos/videos/3/'+str(i)+'.jpg')
    img4 = Image.open('G:/merging photos/videos/4/'+str(i)+'.jpg')
    new_image = Image.new('RGB',(3840,2160))
    new_image.paste(img1,(0,0))
    new_image.paste(img2,(1920,0))  
    new_image.paste(img3,(0,1080))  
    new_image.paste(img4,(1920,1080))
    img = new_image.resize((1920,1080))
    img.save('G:/merging photos//videos/Merged/'+str(i)+'.jpg','JPEG')
    return img

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '-o','--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap1= cv2.VideoCapture('G:/Merging Photos/videos/1.mp4')
    cap2= cv2.VideoCapture('G:/Merging Photos/videos/2.mp4')
    cap3= cv2.VideoCapture('G:/Merging Photos/videos/3.mp4')
    cap4= cv2.VideoCapture('G:/Merging Photos/videos/4.mp4')
    os.mkdir('G:/Merging Photos/videos/1')
    os.mkdir('G:/Merging Photos/videos/2')
    os.mkdir('G:/Merging Photos/videos/3')
    os.mkdir('G:/Merging Photos/videos/4')
    os.mkdir('G:/Merging Photos/videos/Merged')
    fps = None

    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    i = 1
    while(cap1.isOpened() or cap2.isOpened() or cap3.isOpened() or cap4.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        if (ret1 == True and ret2 == True and ret3 == True and ret4 == True):
            cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
            cv2.imwrite('G:/Merging Photos/videos/2/'+str(i)+'.jpg',frame2)
            cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
            cv2.imwrite('G:/Merging Photos/videos/4/'+str(i)+'.jpg',frame4)
            merge(frame1,frame2,frame3,frame4,i)
        
        elif (ret1 == True and ret2 == True and ret3 == True and ret4 == False):
            cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
            cv2.imwrite('G:/Merging Photos/videos/2/'+str(i)+'.jpg',frame2)
            cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
            frame4 = Image.new('RGB',(1920,1080),(0,0,0))
            frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
            merge(frame1,frame2,frame3,frame4,i)
        
        elif (ret1 == True and ret2 == False and ret3 == True and ret4 == False):
            cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
            cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
            frame2 = Image.new('RGB',(1920,1080),(0,0,0))
            frame2.save('G:/merging photos//videos/2/'+str(i)+'.jpg','JPEG')
            frame4 = Image.new('RGB',(1920,1080),(0,0,0))
            frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
            merge(frame1,frame2,frame3,frame4,i)
            
        elif(ret1 == True):
            cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
            frame2 = Image.new('RGB',(1920,1080),(0,0,0))
            frame2.save('G:/merging photos//videos/2/'+str(i)+'.jpg','JPEG')
            frame3 = Image.new('RGB',(1920,1080),(0,0,0))
            frame3.save('G:/merging photos//videos/3/'+str(i)+'.jpg','JPEG')
            frame4 = Image.new('RGB',(1920,1080),(0,0,0))
            frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
            merge(frame1,frame2,frame3,frame4,i)
        
        else:
            break
            
        i+=1
        
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()