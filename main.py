import horizontal_main
import vertical_main
from serial_filter import batch_kuwei_key_frame_filter, batch_kuwei_key_frame_filter_protocol
import parameters as param
from utils.file_util import clear_folder
from utils.report_util import measurement_kuwei_projection

def main(data_src_dir, data_dst_dir):
    batch_kuwei_key_frame_filter(img_dir, data_src_dir)
    horizontal_main.batch_serial_measurement(data_src_dir, data_dst_dir)
    vertical_main.batch_serial_measurement(data_src_dir, data_dst_dir)
    print("[END INFO] Measurement result in the directory: ", data_dst_dir)


def main_protocol(data_src_dir, data_dst_dir, xls_file):
    batch_kuwei_key_frame_filter_protocol(img_dir, data_src_dir)
    horizontal_main.batch_serial_measurement_protocol(data_src_dir, data_dst_dir)
    vertical_main.batch_serial_measurement_protocol(data_src_dir, data_dst_dir)
    print("[WORK FLOW] Generate measurement excel report.")
    measurement_kuwei_projection(data_src_dir, data_dst_dir, xls_file)
    print("[WORK FLOW] Report generated successfully.")
    print("[END INFO] Measurement result in the directory: ", data_dst_dir)
    print("[END INFO] Measurement result report in the directory: ", xls_file)


if __name__ == '__main__':
    print("[START INFO] Start measurement work flow.")
    parser = param.get_parser_for_measurement()
    args = parser.parse_args()

    img_dir = args.img_dir
    data_src_dir = args.src_dir
    data_dst_dir = args.dst_dir
    use_protocol = args.use_protocol
    xls_file = args.xls_file

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

    print("[WORK FLOW] Clear previous result.")
    clear_folder(data_src_dir)
    clear_folder(data_dst_dir)
    
    if not use_protocol:
        if args.kuwei_type == param.KUWEI_TYPE_3:
            main(data_src_dir, data_dst_dir)
    elif use_protocol:
        main_protocol(data_src_dir, data_dst_dir, xls_file)