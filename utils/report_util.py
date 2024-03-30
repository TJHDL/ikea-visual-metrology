import os
import xlrd
from xlutils.copy import copy
import time

def get_kuwei_info_from_images(folder_path):
    image_names = os.listdir(folder_path)
    image_names.sort()

    huojia, floor = folder_path.split('/')[-1].split('_')[0], folder_path.split('/')[-1].split('_')[1]

    kuwei_list = []
    for image_name in image_names:
        kuwei = image_name.split('.')[0].split('_')[-1]
        if not kuwei_list.__contains__(kuwei):
            kuwei_list.append(kuwei)
    
    return huojia, floor, kuwei_list


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
    print("Excel valid rows: ", valid_rows)

    return valid_rows


def edit_report(xls_file, sheet, new_book, huojia, floor, kuwei, safe, valid_rows):
    for idx in range(1, valid_rows, 1):
        if sheet.cell_value(idx, 0) == huojia and sheet.cell_value(idx, 1) == kuwei and sheet.cell_value(idx, 2) == floor:
            current_timestamp = time.time()
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp))
            new_book.get_sheet(0).write(idx, 9, formatted_time)  # 将拍摄时间输入到表格
            if safe:
                new_book.get_sheet(0).write(idx, 10, "正常")
            elif not safe:
                new_book.get_sheet(0).write(idx, 10, "放置异常")
    new_book.save(xls_file)


if __name__ == '__main__':
    pass