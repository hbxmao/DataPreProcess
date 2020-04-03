import os
import cv2
import XMLPreProcess as xpp
import copy
import MakeTxt as mt
import VOC_Label as vocl


# 读取所有图片
def read_dir(in_str, input_images_str, input_xmls_str):

    for index, filename in enumerate(os.listdir(in_str)):
        # print(index)
        if index % 2 == 0:
            input_images_str.append(filename)
        else:
            input_xmls_str.append(filename)

    input_images_str.sort(key=custom_key)
    input_xmls_str.sort(key=custom_key)

    # why fail
    # input_images.sort(key=functools.cmp_to_key(cmp_by_filename))
    # input_XMLs.sort(key=functools.cmp_to_key(cmp_by_filename))
    return


def write_dir(out_str, out_list, extension_str):
    if extension_str.find('bmp') != -1:
        for sub_index in range(len(out_list)):
            out_img = cv2.cvtColor(out_list[sub_index][0], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(out_str + str(sub_index).zfill(6) + '.jpg', out_img)
            # print(sub_index, out_list[sub_index][1:])
    else:
        for sub_index in range(len(out_list)):
            xpp.write_xml(out_str + str(sub_index).zfill(6) + extension_str, out_list[sub_index])

    return


def get_xml_bndbox(et_root, out_bndboxs):
    Object = et_root.findall('object')
    # print('num bndboxs: ', len(Object))

    for i in range(len(Object)):
        cur_bndbox = (Object[i].find('name').text,
                      int(Object[i].find('bndbox').find('ymin').text), int(Object[i].find('bndbox').find('xmin').text),
                      int(Object[i].find('bndbox').find('ymax').text), int(Object[i].find('bndbox').find('xmax').text))
        out_bndboxs.append(cur_bndbox)

    # print('all bndboxs: ', out_bndboxs)
    # print('all bndboxs data: ', out_bndboxs[0][1:])
    return


def get_new_xml(in_target_size, in_ori_et_root, in_bndboxs, in_crop_rect):
    # print(in_crop_rect)
    out_et_root = copy.deepcopy(in_ori_et_root)

    # 改size
    size = out_et_root.find('size')
    size.find('height').text = str(in_target_size[0])
    size.find('width').text = str(in_target_size[1])

    # 删除所有object node
    Object = out_et_root.findall('object')

    if len(Object) > 0:
        Object_4_insert = copy.deepcopy(Object[0])
    else:
        return out_et_root, 0

    del_parent_nodes = out_et_root.getroot().findall('object')
    for node in del_parent_nodes:
        out_et_root.getroot().remove(node)

    contained_bndboxs = []
    for cur_in_bndboxs in in_bndboxs:
        inter_area, rt_and = calc_inter(cur_in_bndboxs[1:], in_crop_rect)
        if inter_area > 0.1:
            defect_rt_cropped = (cur_in_bndboxs[0], rt_and[0] - in_crop_rect[0], rt_and[1] - in_crop_rect[1],
                                 rt_and[2] - in_crop_rect[0], rt_and[3] - in_crop_rect[1])
            contained_bndboxs.append(defect_rt_cropped)
            # print(rt_and, in_crop_rect, defect_rt_cropped)

    # 插入修改后的object nodes
    for cur_out_bndboxs in contained_bndboxs:
        cur_object_4_insert = copy.deepcopy(Object_4_insert)
        cur_object_4_insert.find('name').text = cur_out_bndboxs[0]  # 修改节点名
        cur_object_4_insert.find('bndbox').find('ymin').text = str(cur_out_bndboxs[1])  # 修改节点文本
        cur_object_4_insert.find('bndbox').find('xmin').text = str(cur_out_bndboxs[2])
        cur_object_4_insert.find('bndbox').find('ymax').text = str(cur_out_bndboxs[3])
        cur_object_4_insert.find('bndbox').find('xmax').text = str(cur_out_bndboxs[4])
        out_et_root.getroot().append(cur_object_4_insert)

    xml_str = xpp.tostring(out_et_root.getroot(), encoding='utf8', method='xml')

    xml_str = xml_str.replace('\t</annotation>'.encode(encoding='utf8'), '</annotation>'.encode(encoding='utf8'))

    out_et_root = xpp.ElementTree(xpp.fromstring(xml_str))

    if len(contained_bndboxs) > 0:
        return out_et_root, len(contained_bndboxs)
    else:
        return out_et_root, 0


def calc_inter(rec1, rec2):
    """
    :param rec1: (x0,y0,x1,y1) in opencv
    :param rec2:(x0,y0,x1,y1)
    :return:相交面积除rec1面积.
    """
    left_column_max = max(rec1[1], rec2[1])
    right_column_min = min(rec1[3], rec2[3])
    up_row_max = max(rec1[0], rec2[0])
    down_row_min = min(rec1[2], rec2[2])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0, ()
        # 两矩形有相交区域的情况
    else:
        s1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        s2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        s_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return s_cross / (s1 * 1.0), (up_row_max, left_column_max, down_row_min, right_column_min)


# 按指定大小切割图片
def split_img(in_img_path, in_xml_path, out_imgs, out_xmls, out_neg_imgs, out_neg_xmls, target_size):
    # image
    cur_img = cv2.imread(in_img_path, cv2.IMREAD_UNCHANGED)
    target_rows = target_size[0]
    target_cols = target_size[1]

    # xml
    cur_xml = xpp.read_xml(in_xml_path)
    cur_bndboxs = []
    get_xml_bndbox(cur_xml, cur_bndboxs)

    # print(target_size, cur_img.shape)

    x0 = y0 = 0
    x1 = target_rows-1
    y1 = target_cols-1
    row_count = cur_img.shape[0] // target_rows
    col_count = cur_img.shape[1] // target_cols
    # print(row_count, col_count)

    # loop gen sub images
    for x in range(row_count):
        x0_0 = x0 + x * target_rows
        x1_1 = x1 + x * target_rows
        for y in range(col_count):
            y0_0 = y0 + y * target_cols
            y1_1 = y1 + y * target_cols
            # print('point: ', x0_0, y0_0, x1_1, y1_1)
            out_xml, defects_count = get_new_xml(target_size, cur_xml, cur_bndboxs, (x0_0, y0_0, x1_1, y1_1))
            if defects_count > 0:
                out_imgs.append((cur_img[x0_0:x1_1+1, y0_0:y1_1+1], x0_0, y0_0, x1_1, y1_1))
                out_xmls.append(out_xml)
            else:
                out_neg_imgs.append((cur_img[x0_0:x1_1 + 1, y0_0:y1_1 + 1], x0_0, y0_0, x1_1, y1_1))
                out_neg_xmls.append(out_xml)

    # last row
    x0 = cur_img.shape[0] - 1 - target_rows
    y0 = 0
    x1 = cur_img.shape[0] - 1
    y1 = target_cols - 1
    for y in range(col_count):
        y0_0 = y0 + y * target_cols
        y1_1 = y1 + y * target_cols
        # print('last row: ', x0, y0_0, x1, y1_1)
        out_xml, defects_count = get_new_xml(target_size, cur_xml, cur_bndboxs, (x0, y0_0, x1, y1_1))
        if defects_count > 0:
            out_imgs.append((cur_img[x0:x1, y0_0:y1_1 + 1], x0, y0_0, x1, y1_1))
            out_xmls.append(out_xml)
        else:
            out_neg_imgs.append((cur_img[x0:x1, y0_0:y1_1 + 1], x0, y0_0, x1, y1_1))
            out_neg_xmls.append(out_xml)

    # last col
    x0 = 0
    y0 = cur_img.shape[1] - 1 - target_cols
    x1 = target_rows - 1
    y1 = cur_img.shape[1] - 1
    for x in range(row_count):
        x0_0 = x0 + x * target_rows
        x1_1 = x1 + x * target_rows
        # print('last col: ', x0_0, y0, x1_1, y1)
        out_xml, defects_count = get_new_xml(target_size, cur_xml, cur_bndboxs, (x0_0, y0, x1_1, y1))
        if defects_count > 0:
            out_imgs.append((cur_img[x0_0:x1_1 + 1, y0:y1], x0_0, y0, x1_1, y1))
            out_xmls.append(out_xml)
        else:
            out_neg_imgs.append((cur_img[x0_0:x1_1 + 1, y0:y1], x0_0, y0, x1_1, y1))
            out_neg_xmls.append(out_xml)

    # last corner
    x0 = cur_img.shape[0] - 1 - target_rows
    y0 = cur_img.shape[1] - 1 - target_cols
    x1 = cur_img.shape[0] - 1
    y1 = cur_img.shape[1] - 1
    # print('right bottom corner: ', x0, y0, x1, y1)
    out_xml, defects_count = get_new_xml(target_size, cur_xml, cur_bndboxs, (x0, y0, x1, y1))
    if defects_count > 0:
        out_imgs.append((cur_img[x0:x1, y0:y1], x0, y0, x1, y1))
        out_xmls.append(out_xml)
    else:
        out_imgs.append((cur_img[x0:x1, y0:y1], x0, y0, x1, y1))
        out_neg_xmls.append(out_xml)

    return


def cmp_by_filename(str1, str2):
    idx1 = int(str1[:str1.find('.')])
    idx2 = int(str2[:str2.find('.')])
    return idx1 < idx2


def custom_key(cru_str):
    return int(cru_str[:cru_str.find('.')])


def main():
    input_folder_str = 'D:/Cosmetic/stage1/20200317_yaohua_huaheng'
    out_folder_str = 'D:/Cosmetic/stage1/split'
    input_images = []
    input_xmls = []

    # read all images and xmls
    read_dir(input_folder_str, input_images, input_xmls)
    if len(input_images) != len(input_xmls):
        return -1
    print(input_images, '\n', input_xmls)

    # do split both on image and xml
    out_images = []
    out_xmls = []
    target_size = (416, 416)
    for index in range(len(input_images)):  # len(input_images)
        split_img(input_folder_str + '/' + input_images[index],
                  input_folder_str + '/' + input_xmls[index],
                  out_images, out_xmls, target_size)
        print(len(out_images))

        # write to target folder
        write_dir(out_folder_str + '/imgs/' + str(index) + '_', out_images, '.bmp')
        write_dir(out_folder_str + '/xmls/' + str(index) + '_', out_xmls, '.xml')

        out_images.clear()
        out_xmls.clear()
    return


def main_root():
    root_input_str = 'D:/Cosmetic/stage1/'
    out_folder_str = 'D:/Cosmetic/stage1/split'
    out_images = []
    out_xmls = []
    out_neg_images = []
    out_neg_xmls = []
    target_size = (608, 608)

    for root_index, root_filename in enumerate(os.listdir(root_input_str)):
        print(root_filename)
        # if root_index > 0:
        #     break
        if not root_filename.strip() or not root_filename[0].isnumeric():
            continue

        input_folder_str = root_input_str + root_filename  # 'D:/Cosmetic/stage1/20200317_yaohua_huaheng'
        # print(input_folder_str)

        input_images = []
        input_xmls = []
        # read all images and xmls
        read_dir(input_folder_str, input_images, input_xmls)
        if len(input_images) != len(input_xmls):
            print('Error: '+ input_folder_str + ' has wrong file nums')
            continue
        # print(input_images, '\n', input_xmls)
        for index in range(len(input_images)):  # len(input_images)
            split_img(input_folder_str + '/' + input_images[index],
                      input_folder_str + '/' + input_xmls[index],
                      out_images, out_xmls, out_neg_images, out_neg_xmls, target_size)
            print(index, ', ', end="")
        print('\n')

    if len(out_images) != len(out_xmls):
        print('Error: out_images & out_xmls')
        return -1

    print('start writing: ')
    write_dir(out_folder_str + '/imgs/', out_images, '.bmp')
    write_dir(out_folder_str + '/xmls/', out_xmls, '.xml')

    print('done split!')

    return 1


# main
if __name__ == '__main__':
    # main_root()
    # mt.make_txt()
    vocl.VOC_Label()

