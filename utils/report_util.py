import os
import re
import xlrd
from xlutils.copy import copy
import time
import parameters as param
from utils import get_file_description, close_file_description

xls_file = r'report\407-03-00-60_20231129.xls'

def get_kuwei_info_from_images(folder_path):
    image_names = os.listdir(folder_path)
    image_names.sort()

    huojia, floor = folder_path.split('\\')[-1].split('_')[0], folder_path.split('/')[-1].split('_')[1]
    kuwei_type = param.KUWEI_TYPE_2 if len(os.listdir(folder_path)) <= param.KUWEI_TYPE_IMAGES_NUM_THRESHOLD['2-3'] else param.KUWEI_TYPE_3
    kuwei_type = param.KUWEI_TYPE_4 if len(os.listdir(folder_path)) >= param.KUWEI_TYPE_IMAGES_NUM_THRESHOLD['3-4'] else kuwei_type

    kuwei_list = []
    for image_name in image_names:
        kuwei = image_name.split('.')[0].split('_')[-1]
        if not kuwei_list.__contains__(kuwei):
            kuwei_list.append(kuwei)
    
    return huojia, floor, kuwei_list, kuwei_type


def get_xls_workbook_sheet(xls_file):
    workbook = xlrd.open_workbook(xls_file, formatting_info=False)
    sheet = workbook.sheet_by_index(0)  # 假设只有一个工作表

    # 设置表头
    new_book = copy(workbook)
    new_book.get_sheet(0).write(0, 9, '盘点时间')
    new_book.get_sheet(0).write(0, 10, '盘点结果')

    return workbook, sheet, new_book


def count_xls_valid_rows(sheet):
    valid_rows = sum(1 for row in range(sheet.nrows) if any(sheet.cell(row, col).value for col in range(sheet.ncols)))
    print("[INFO] Excel valid rows: ", valid_rows)

    return valid_rows


def get_txt_key_row_number(fd):
    key_row_dict = {"horizontal_gap":-1, "horizontal_box":-1, "vertical_gap":-1}

    lines = fd.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()

        if re.search("横向间隙尺寸", line):
            key_row_dict["horizontal_gap"] = idx
        elif re.search("横向货物尺寸", line):
            key_row_dict["horizontal_box"] = idx
        elif re.search("纵向间隙尺寸", line):
            key_row_dict["vertical_gap"] = idx
    
    return key_row_dict


def get_txt_measurement_result(fd, kuwei_type, key_row_dict):
    if kuwei_type == param.KUWEI_TYPE_3:
        horizontal_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY, "gap3":param.POSITIVE_INFINITY, "gap4":param.POSITIVE_INFINITY}
        vertical_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY, "gap3":param.POSITIVE_INFINITY}
    elif kuwei_type == param.KUWEI_TYPE_2:
        horizontal_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY, "gap3":param.POSITIVE_INFINITY}
        vertical_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY}
    elif kuwei_type == param.KUWEI_TYPE_4:
        horizontal_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY, "gap3":param.POSITIVE_INFINITY, "gap4":param.POSITIVE_INFINITY, "gap5":param.POSITIVE_INFINITY}
        vertical_size_dict = {"gap1":param.POSITIVE_INFINITY, "gap2":param.POSITIVE_INFINITY, "gap3":param.POSITIVE_INFINITY, "gap4":param.POSITIVE_INFINITY}
    else:
        print("[ERROR] Unknown kuwei type!!!")
        raise Exception("Kuwei type invalid!")
    
    lines = fd.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()

        if key_row_dict['horizontal_gap'] != -1 and idx > key_row_dict['horizontal_gap'] and idx < key_row_dict['vertical_gap']:
            if re.search("间隙1", line):
                horizontal_size_dict["gap1"] = float(line.split(':')[-1])
            elif re.search("间隙2", line):
                horizontal_size_dict["gap2"] = float(line.split(':')[-1])
            elif re.search("间隙3", line):
                horizontal_size_dict["gap3"] = float(line.split(':')[-1])
            elif re.search("间隙4", line):
                horizontal_size_dict["gap4"] = float(line.split(':')[-1])
            elif re.search("间隙5", line):
                horizontal_size_dict["gap5"] = float(line.split(':')[-1])
        elif key_row_dict['vertical_gap'] != -1 and idx > key_row_dict['vertical_gap']:
            if re.search("间隙1", line):
                vertical_size_dict["gap1"] = float(line.split(':')[-1])
            elif re.search("间隙2", line):
                vertical_size_dict["gap2"] = float(line.split(':')[-1])
            elif re.search("间隙3", line):
                vertical_size_dict["gap3"] = float(line.split(':')[-1])
            elif re.search("间隙4", line):
                vertical_size_dict["gap4"] = float(line.split(':')[-1])
    
    return horizontal_size_dict, vertical_size_dict


def kuwei_safe_judge(kuwei_number, horizontal_size_dict, vertical_size_dict):
    if kuwei_number == 1:
        if horizontal_size_dict['gap1'] < param.HORIZONTAL_SAFE_THRESHOLD or horizontal_size_dict['gap2'] < param.HORIZONTAL_SAFE_THRESHOLD \
            or vertical_size_dict['gap1'] < param.VERTICAL_SAFE_THRESHOLD:
            return False
    elif kuwei_number == 2:
        if horizontal_size_dict['gap2'] < param.HORIZONTAL_SAFE_THRESHOLD or horizontal_size_dict['gap3'] < param.HORIZONTAL_SAFE_THRESHOLD \
            or vertical_size_dict['gap2'] < param.VERTICAL_SAFE_THRESHOLD:
            return False
    elif kuwei_number == 3:
        if horizontal_size_dict['gap3'] < param.HORIZONTAL_SAFE_THRESHOLD or horizontal_size_dict['gap4'] < param.HORIZONTAL_SAFE_THRESHOLD \
            or vertical_size_dict['gap3'] < param.VERTICAL_SAFE_THRESHOLD:
            return False
    elif kuwei_number == 4:
        if horizontal_size_dict['gap4'] < param.HORIZONTAL_SAFE_THRESHOLD or horizontal_size_dict['gap5'] < param.HORIZONTAL_SAFE_THRESHOLD \
            or vertical_size_dict['gap4'] < param.VERTICAL_SAFE_THRESHOLD:
            return False
    
    return True

def judge_safe_dict(horizontal_size_dict, vertical_size_dict, kuwei_type):
    safe_list = []
    
    if kuwei_type == param.KUWEI_TYPE_3:
        safe_list.append(kuwei_safe_judge(3, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(2, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(1, horizontal_size_dict, vertical_size_dict))
    elif kuwei_type == param.KUWEI_TYPE_2:
        safe_list.append(kuwei_safe_judge(2, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(1, horizontal_size_dict, vertical_size_dict))
    elif kuwei_type == param.KUWEI_TYPE_4:
        safe_list.append(kuwei_safe_judge(4, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(3, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(2, horizontal_size_dict, vertical_size_dict))
        safe_list.append(kuwei_safe_judge(1, horizontal_size_dict, vertical_size_dict))
    else:
        print("[ERROR] Unknown kuwei type!!!")
        raise Exception("Kuwei type invalid!")

    return safe_list


def edit_report(xls_file, sheet, new_book, huojia, floor, kuwei, safe, valid_rows):
    for idx in range(1, valid_rows, 1):
        if str(sheet.cell_value(idx, 0)) == huojia and str(sheet.cell_value(idx, 1)) == kuwei and str(sheet.cell_value(idx, 2)) == floor:
            current_timestamp = time.time()
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp))
            new_book.get_sheet(0).write(idx, 9, formatted_time)  # 将修改时间输入到表格
            
            if safe is None:
                new_book.get_sheet(0).write(idx, 10, "测量失败")
                continue

            if safe:
                new_book.get_sheet(0).write(idx, 10, "正常")
            elif not safe:
                new_book.get_sheet(0).write(idx, 10, "放置异常")
    new_book.save(xls_file)


def measurement_kuwei_projection(img_dir, data_dst_dir, xls_file):
    workbook, sheet, new_book = get_xls_workbook_sheet(xls_file)
    valid_rows = count_xls_valid_rows(sheet)

    src_dirs = os.listdir(img_dir)
    for src_dir in src_dirs:
        huojia, floor, kuwei_list, kuwei_type = get_kuwei_info_from_images(os.path.join(img_dir, src_dir))
        if not os.path.exists(os.path.join(data_dst_dir, src_dir + '_' + str(kuwei_type))):
            for idx, kuwei in enumerate(kuwei_list):
                edit_report(xls_file, sheet, new_book, huojia, floor, kuwei, None, valid_rows)
            print("[WARNING] Measurement result is empty: ", os.path.join(data_dst_dir, src_dir + '_' + str(kuwei_type)))
            continue
        fd = get_file_description(os.path.join(data_dst_dir, src_dir + '_' + str(kuwei_type)), "measurement.txt")
        key_row_dict = get_txt_key_row_number(fd)
        horizontal_size_dict, vertical_size_dict = get_txt_measurement_result(fd, kuwei_type, key_row_dict)
        safe_list = judge_safe_dict(horizontal_size_dict, vertical_size_dict, kuwei_type)

        for idx, kuwei in enumerate(kuwei_list):
            edit_report(xls_file, sheet, new_book, huojia, floor, kuwei, safe_list[idx], valid_rows)
        close_file_description(fd)


if __name__ == '__main__':
    pass