import horizontal_main
import vertical_main
from serial_filter import batch_kuwei_key_frame_filter, batch_kuwei_key_frame_filter_protocol
import parameters as param
from utils.file_util import clear_folder
from utils.report_util import measurement_kuwei_projection, get_xls_workbook_sheet, count_xls_valid_rows, edit_report
from utils.marking_points_util import marking_points_test
from utils.semantic_info_util import LEDNet_test

def measurement_main(img_dir, data_src_dir, data_dst_dir):
    batch_kuwei_key_frame_filter(img_dir, data_src_dir)
    horizontal_main.batch_serial_measurement(data_src_dir, data_dst_dir)
    vertical_main.batch_serial_measurement(data_src_dir, data_dst_dir)
    print("[END INFO] Measurement result in the directory: ", data_dst_dir)


def measurement_main_protocol(img_dir, data_src_dir, data_dst_dir, xls_file):
    batch_kuwei_key_frame_filter_protocol(img_dir, data_src_dir)
    horizontal_main.batch_serial_measurement_protocol(data_src_dir, data_dst_dir)
    vertical_main.batch_serial_measurement_protocol(data_src_dir, data_dst_dir)
    print("[WORK FLOW] Generate measurement excel report.")
    measurement_kuwei_projection(img_dir, data_dst_dir, xls_file)
    print("[WORK FLOW] Report generated successfully.")
    print("[END INFO] Measurement result in the directory: ", data_dst_dir)
    print("[END INFO] Measurement result report in the directory: ", xls_file)


def workflow_main():
    print("[START INFO] Start measurement work flow.")
    parser = param.get_parser_for_measurement()
    args = parser.parse_args()

    img_dir = args.img_dir
    data_src_dir = args.src_dir
    data_dst_dir = args.dst_dir
    use_protocol = args.use_protocol
    xls_file = args.xls_file
    param.MODEL_MODE = args.model_mode

    if args.floor:
        param.FLOOR_NUM = args.floor
        param.H_CAMERA = param.FLOOR_NUM * param.FLOOR_HEIGHT - (param.CAR_HEIGHT + param.UAV_HEIGHT[param.FLOOR_NUM]) - param.TIEPIAN_WIDTH

    print("[INPUT ARGUMENTS] img_dir: ", img_dir)
    print("[INPUT ARGUMENTS] data_src_dir: ", data_src_dir)
    print("[INPUT ARGUMENTS] data_dst_dir: ", data_dst_dir)
    print("[INPUT ARGUMENTS] floor: ", param.FLOOR_NUM)
    print("[INPUT ARGUMENTS] camera_height: ", param.H_CAMERA)
    print("[INPUT ARGUMENTS] use_protocol: ", use_protocol)
    print("[INPUT ARGUMENTS] xls_file: ", xls_file)
    print("[INPUT ARGUMENTS] model_mode: ", param.MODEL_MODE)

    print("[WORK FLOW] Clear previous result.")
    clear_folder(data_src_dir)
    clear_folder(data_dst_dir)
    
    if not use_protocol:
        print("[INFO] Not use protocol test.")
        if args.kuwei_type == param.KUWEI_TYPE_3:
            measurement_main(img_dir, data_src_dir, data_dst_dir)
    elif use_protocol:
        print("[INFO] Use protocol test.")
        measurement_main_protocol(img_dir, data_src_dir, data_dst_dir, xls_file)


def marking_points_test_main():
    marking_points_test(r'C:\Users\95725\Desktop\rtsp_picture_20240322\floor3', r'C:\Users\95725\Desktop\rtsp_picture_20240322\floor3_points')


def LEDNet_test_main():
    LEDNet_test(r'C:\Users\95725\Desktop\rtsp_picture_1009_407_floor3', r'C:\Users\95725\Desktop\semantic')


if __name__ == '__main__':
    workflow_main()
    # marking_points_test_main()
    # LEDNet_test_main()