import os
import sys

import cv2

from tqdm import tqdm
import numpy as np

import tarfile
from util.multithread_video_writer import MultithreadVideoWriter

def extract_all_tar(root_dir="."):
    for file in os.listdir(root_dir):
        if file.endswith(".tar"):
            tar_path = os.path.join(root_dir, file)

            # 生成同名文件夹
            folder_name = os.path.basename(os.path.normpath(root_dir))
            extract_path = os.path.join(root_dir, folder_name)

            print(f"Processing: {file}")

            # 创建目录
            os.makedirs(extract_path, exist_ok=True)

            # 解压
            try:
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=extract_path)
                print(f"Extracted to: {extract_path}")

                # 删除 tar 文件
                #os.remove(tar_path)
                #print(f"Deleted: {file}")

            except Exception as e:
                print(f"Failed: {file}, Error: {e}")

def main():
    saved_dir = './saved_images/run_video/'
    tar_dirs = get_subfolders(saved_dir)
    tar_dirs.sort()
    print(tar_dirs)
    for tar_dir in tar_dirs:
        print('-------------------------------')
        print("Processing tar folder: {}".format(tar_dir))
        mask_name = os.path.basename(os.path.normpath(tar_dir))
        if os.path.isdir(os.path.join(tar_dir,mask_name)):
            print("mask already generated")
        else:
            print("generating mask")
            process_tar_dir(tar_dir)
        if os.path.exists(os.path.join(tar_dir,mask_name+'_mask.mp4')):
            print("mask video already generated")
        else:
            print("generating mask video")
            generate_mask_video(tar_dir)

def generate_mask_video(tar_dir):
    mask_name = os.path.basename(os.path.normpath(tar_dir))
    assert os.path.isdir(os.path.join(tar_dir,mask_name))
    img_list = get_file_path_list(os.path.join(tar_dir,mask_name),'png')
    video_writer=MultithreadVideoWriter(os.path.join(tar_dir, mask_name+'_mask.mp4'))
    for i in tqdm(range(len(img_list))):
        #if i>=len(video_loader0)*0.66:
        #    break
        data=cv2.imread(img_list[i],cv2.IMREAD_UNCHANGED)
        frame = data[:,:,:3]
        mask = data[:,:,3]
        frame[mask==0] = np.array([255,255,255],np.uint8)
        video_writer.append(frame,False)
    video_writer.make_video()
    video_writer.close()






def process_tar_dir(dir_path):
    extract_all_tar(dir_path)
    root_dir = dir_path
    mask_name = os.path.basename(os.path.normpath(dir_path))
    img_dirs = get_subfolders(root_dir)
    #print(img_dirs)
    if len(img_dirs) == 1:
        print("Nothing to do, just rename")
        os.rename(img_dirs[0],os.path.join(root_dir,mask_name))
        return
    img_lists = [get_file_path_list(img_dir,'png') for img_dir in img_dirs]
    for img_list in img_lists:
        #print(img_list[0])
        print(len(img_list))
    folder_name = os.path.basename(os.getcwd())
    target_dir = os.path.join(root_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    for i in tqdm(range(len(img_lists[0]))):
        #if i>=len(video_loader0)*0.66:
        #    break
        datas=[cv2.imread(img_list[i],cv2.IMREAD_UNCHANGED) for img_list in img_lists]
        frame = mask_overlay(datas)
        frame_path = os.path.join(target_dir,str(i).zfill(8)+'.png')
        cv2.imwrite(frame_path,frame)



def mask_overlay(images):
    """
    images: list of RGBA images (H, W, 4), dtype uint8
    order: first = bottom layer
    """

    out = images[0].copy()

    for img in images[1:]:
        mask = img[..., 3] > 0
        out[mask] = img[mask]
    return out

def get_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path)
                  if f.is_dir() and not f.name.startswith('.')]
    return subfolders

def get_file_path_list(dir_path:str,type_list=None)->list:
    filelist = os.listdir(dir_path)
    filelist.sort()
    file_list = []



    for item in filelist:
        if check_type(item,type_list):
            if item.startswith('.'):
                continue
            # print(item)
            # print(item.split('.')[0])
            file_list.append(item)
    file_path_list=[os.path.join(dir_path,item) for item in file_list]
    return file_path_list


def check_type(path,type_list):
    if type_list is None:
        return True
    if not isinstance(type_list, list):
        type_list = [type_list]
    result=False
    for i in range(len(type_list)):
        if path.endswith(type_list[i]):
            result=True
            break
    return result



if __name__ == "__main__":
    main()