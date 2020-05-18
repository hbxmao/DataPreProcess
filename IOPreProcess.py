import os
import XMLPreProcess as xpp
import cv2
import copy

def read_all_path(in_path, out_paths_array=[]):
    """
    :param out_paths_array: out array of item names in in_path
    :param in_path: root path of listdir
    :return: read_success: read success or not
             out_paths_array: contains all item paths
    """
    read_success = True
    print('reading items from in_path: ' + in_path)
    for root_index, root_filename in enumerate(os.listdir(in_path)):
        # print('cur item:' + root_filename)
        if not root_filename.strip() or not root_filename[root_filename.find('.')-1].isnumeric():
            print('existing illegal item')
            read_success = False
            break
        out_paths_array.append(in_path + '/' + root_filename)

    return read_success


def create_all_sub_roi_rects(image_size, target_size):
    all_image_sub_rects = []

    # in file sys coord starts from col, in cv starts from row
    target_rows = target_size[0]
    target_cols = target_size[1]
    ori_rows = image_size[0]
    ori_cols = image_size[1]
    row_count = ori_rows // target_rows
    col_count = ori_cols // target_cols

    x0 = y0 = 0
    x1 = target_size[0] - 1
    y1 = target_size[1] - 1

    # part 1
    for x in range(row_count):
        x0_0 = x0 + x * target_rows
        x1_1 = x1 + x * target_rows
        for y in range(col_count):
            y0_0 = y0 + y * target_cols
            y1_1 = y1 + y * target_cols
            all_image_sub_rects.append((x0_0, y0_0, x1_1, y1_1))

    # last row
    x0 = ori_rows - 1 - target_rows
    y0 = 0
    x1 = ori_rows - 1
    y1 = target_cols - 1
    for y in range(col_count):
        y0_0 = y0 + y * target_cols
        y1_1 = y1 + y * target_cols
        all_image_sub_rects.append((x0, y0_0, x1, y1_1))

    # last col
    x0 = 0
    y0 = ori_cols - 1 - target_cols
    x1 = target_rows - 1
    y1 = ori_cols - 1
    for x in range(row_count):
        x0_0 = x0 + x * target_rows
        x1_1 = x1 + x * target_rows
        all_image_sub_rects.append((x0_0, y0, x1_1, y1))

    # last corner
    x0 = ori_rows - 1 - target_rows
    y0 = ori_cols - 1 - target_cols
    x1 = ori_rows - 1
    y1 = ori_cols - 1
    all_image_sub_rects.append((x0, y0, x1, y1))

    return all_image_sub_rects

def get_xml_bndbox(et_root, out_bndboxs):
    Object = et_root.findall('object')
    # print('num bndboxs: ', len(Object))

    # in xml file, boundingboxs width goes first, so we get ymin first, because sub rects height goes first
    for i in range(len(Object)):
        cur_bndbox = (Object[i].find('name').text,
                      int(Object[i].find('bndbox').find('ymin').text), int(Object[i].find('bndbox').find('xmin').text),
                      int(Object[i].find('bndbox').find('ymax').text), int(Object[i].find('bndbox').find('xmax').text))
        out_bndboxs.append(cur_bndbox)

    # print('all bndboxs: ', out_bndboxs)
    # print('all bndboxs data: ', out_bndboxs[0][1:])
    return

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

def get_sub_image_classes(in_annotation_bndboxs, cur_sub_rt, out_list_classes):

    for cur_in_bndboxs in in_annotation_bndboxs:
        inter_area, rt_and = calc_inter(cur_in_bndboxs[1:], cur_sub_rt)

        if inter_area > 0.1:
            defect_rt_cropped = (cur_in_bndboxs[0], rt_and[0] - cur_sub_rt[0], rt_and[1] - cur_sub_rt[1],
                                 rt_and[2] - cur_sub_rt[0], rt_and[3] - cur_sub_rt[1])
            out_list_classes.append(defect_rt_cropped)

    return 0

def write_sub_single_sample(in_out_path, in_out_image, in_out_ori_xml, in_sub_rt, in_out_sub_classes):

    cur_img = cv2.imread(in_out_image, cv2.IMREAD_UNCHANGED)
    out_img = cur_img[in_sub_rt[0]:in_sub_rt[2], in_sub_rt[1]:in_sub_rt[3]]
    cv2.imwrite(in_out_path + '.jpg', out_img)

    cur_xml = xpp.read_xml(in_out_ori_xml)
    out_et_root = copy.deepcopy(cur_xml)

    # change size
    size = out_et_root.find('size')
    size.find('height').text = str(out_img.shape[0])
    size.find('width').text = str(out_img.shape[1])

    # change channel num
    size.find('depth').text = '3'

    # delete all object nodes
    Object = out_et_root.findall('object')

    if len(Object) > 0:
        Object_4_insert = copy.deepcopy(Object[0])
    else:
        return out_et_root, 0

    del_parent_nodes = out_et_root.getroot().findall('object')
    for node in del_parent_nodes:
        out_et_root.getroot().remove(node)

    out_classes = []
    # insert modified object nodes
    for cur_out_bndboxs in in_out_sub_classes[2]:
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

    xpp.write_xml(in_out_path + '.xml', out_et_root)

    return 0