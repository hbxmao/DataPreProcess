import os
import cv2
import XMLPreProcess as xpp
import copy
import MakeTxt as mt
import VOC_Label as vocl
import IOPreProcess as iopp
import random
import numpy as np

# classes are targets
# n_classes are items should not be included
classes = ["scratch", "oilstain", "gum"]
n_classes = ["blobpaint"]


def main():
    root_input_path_strs = ['D:/Project_Sources/Cosmetic/stage2/Cosmetic_20200426_scratch_oilstain_gum',
                            'D:/Project_Sources/Cosmetic/stage2/Cosmetic_20200430_scratch_oilstain_gum'
                            # 'D:/Project_Sources/Cosmetic/stage_test',
                            # 'D:/Project_Sources/Cosmetic/stage_test_1'
                            ]
    out_folder_path_str = 'D:/Project_Sources/Cosmetic/stage2_out/CMB'

    # all paths to each image from input path, list index is process index
    in_paths_list = []
    for cur_root in root_input_path_strs:

        read_success = iopp.read_all_path(cur_root, in_paths_list)

        if not read_success:
            raise RuntimeError('root path item reading error')
            break
        else:
            print('done reading ' + cur_root)

    # split items into images and annotations
    if len(in_paths_list) % 2 != 0:
        raise RuntimeError('root path item reading amount error')

    # image is before annotation (bmp > xml)
    in_image_list = in_paths_list[::2]
    in_annotation_list = in_paths_list[1::2]

    print('start sampling...')

    # add relation between sub images and origin image
    cur_img = cv2.imread(in_image_list[0], cv2.IMREAD_UNCHANGED)
    in_size = cur_img.shape
    target_size = (512, 512)

    # get all sub image rects for each image (x0, y0, x1, y1) x for row, y for col
    all_image_sub_rects = iopp.create_all_sub_roi_rects(in_size, target_size)

    # apply sub images to each ori image as its member
    all_image_sub_rt_classes = []
    cur_bndboxs = []

    for id_in_image in range(len(in_image_list)):
        #get xml
        cur_xml = xpp.read_xml(in_annotation_list[id_in_image])
        cur_bndboxs.clear()
        # get all current annotations in xml file
        iopp.get_xml_bndbox(cur_xml, cur_bndboxs)

        for id_sub_rt in range(len(all_image_sub_rects)):
            cur_sub_rt_classes = []
            iopp.get_sub_image_classes(cur_bndboxs, all_image_sub_rects[id_sub_rt], cur_sub_rt_classes)

            # structure: (index of images, index of sub rects, classes of sub rect)
            all_image_sub_rt_classes.append((id_in_image, id_sub_rt, cur_sub_rt_classes))

    # at this point all image has been processed
    # now should start statistic process

    # initial
    all_single_cls_list = []
    for id_s in range(len(classes)):
        single_cls_list = []
        all_single_cls_list.append(single_cls_list)
        #single_cls_list.append((id_s))
    for id_ns in range(len(n_classes)):
        single_ncls_list = []
        all_single_cls_list.append(single_ncls_list)
        #single_ncls_list.append((id_ns+3))

    # loop to fulfill all_single_cls_list
    for id_sub_cls in range(len(all_image_sub_rt_classes)):
        cur_sub_rt = all_image_sub_rt_classes[id_sub_cls]
        cur_sub_classes = cur_sub_rt[2]
        if len(cur_sub_classes) == 0 : continue
        for cur_sub_class in cur_sub_classes:
            if cur_sub_class[0] in classes:
                cur_class_id = classes.index(cur_sub_class[0])
                all_single_cls_list[cur_class_id].append(cur_sub_rt)
            elif cur_sub_class[0] in n_classes:
                cur_nclass_id = n_classes.index(cur_sub_class[0])
                all_single_cls_list[cur_nclass_id + len(classes)].append(cur_sub_rt)

    # print amount of each class
    print('amounts of each class: ')
    length_list = []
    for id_s_l in range(len(all_single_cls_list)):
        cur_len = len(all_single_cls_list[id_s_l])
        if id_s_l < len(classes):
            print(classes[id_s_l], cur_len)
            length_list.append(cur_len)
        else:
            print(n_classes[id_s_l-len(classes)], len(all_single_cls_list[id_s_l]))


    # at this point all single classes has been processed
    # now should start output process
    target_single_cls_amount = min(length_list)
    target_defect_free_amount = target_single_cls_amount * 1 * len(classes)
    print('sampling '+str(target_single_cls_amount)+' for each kind of defects... (n_classes ant included)')
    print('sampling '+str(target_defect_free_amount)+' defects free... ')

    # shuffle all_image_sub_rt_classes
    random.shuffle(all_image_sub_rt_classes)

    length_list.clear()
    num_already_worte = 0
    for i in range(len(classes) + 1):
        length_list.append(0)

    # start finally output
    for id_sub_cls in range(len(all_image_sub_rt_classes)):
        cur_sub_rt_cls = all_image_sub_rt_classes[id_sub_cls]

        # check n_classes
        has_ncls = False
        for cur_cls in cur_sub_rt_cls[2]:
            if cur_cls[0] in n_classes:
                has_ncls = True
                break
        if has_ncls: continue

        # check length
        # update length list
        if len(cur_sub_rt_cls[2]) == 0:
            if length_list[len(classes)] == target_defect_free_amount:
                continue
        else:
            has_enough = False
            for cur_cls in cur_sub_rt_cls[2]:
                cur_id = classes.index(cur_cls[0])
                if length_list[cur_id] == target_single_cls_amount:
                    has_enough = True
                    break
            if has_enough:
                continue

        # start write
        out_image = in_image_list[cur_sub_rt_cls[0]]
        out_xml = in_annotation_list[cur_sub_rt_cls[0]]
        out_sub_rt = all_image_sub_rects[cur_sub_rt_cls[1]]
        iopp.write_sub_single_sample(out_folder_path_str+'/'+ str(num_already_worte).zfill(6),
                                     out_image,
                                     out_xml,
                                     out_sub_rt,
                                     cur_sub_rt_cls
                                     )

        if len(cur_sub_rt_cls[2]) == 0:
            length_list[len(classes)] += 1
        else:
            for cur_cls in cur_sub_rt_cls[2]:
                cur_id = classes.index(cur_cls[0])
                length_list[cur_id] += 1

        num_already_worte += 1

    print('sampled ' + str(num_already_worte) + ' images in total!')
    print('done')

    return 0

# main
if __name__ == '__main__':
    #main()
    mt.make_txt()
    vocl.VOC_Label()