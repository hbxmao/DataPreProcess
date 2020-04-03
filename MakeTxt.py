import os
import random


def make_txt():
    trainval_percent = 0.1
    train_percent = 0.9
    xmlfilepath = 'D:/Cosmetic/stage1/split/data/Annotations'
    txtsavepath = 'D:/Cosmetic/stage1/split/data/ImageSets'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open('D:/Cosmetic/stage1/split/data/ImageSets/trainval.txt', 'w')
    ftest = open('D:/Cosmetic/stage1/split/data/ImageSets/test.txt', 'w')
    ftrain = open('D:/Cosmetic/stage1/split/data/ImageSets/train.txt', 'w')
    fval = open('D:/Cosmetic/stage1/split/data/ImageSets/val.txt', 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:
            ftrain.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    return
