from collections import deque
import multiprocessing
import serial
import time
from matplotlib import cm
import numpy as np

def clear_screen():
    print("\033[H\033[J", end='')


def serial_reader(done_flag: multiprocessing.Value, data_queue: multiprocessing.Queue, sample_time: float = 1/100):
    ser = serial.Serial(
        port='/dev/tty.usbmodem1103',
        baudrate=921600,
        timeout=1
    )

    if ser.isOpen():
        print(f"Serial port {ser.port} is open.")
    else:
        print(f"Failed to open serial port {ser.port}.")
        exit()

    total_time_start = time.time()


    print("RESET DEVICE TO BEGIN")
    while True:
        data = ser.readline()
        try:
            data_str = data.decode().strip()
            print(f"\033[38;5;{51}m{data_str}\033[0m")
            if "DATA_BEGIN" in data_str:
                break
        except UnicodeDecodeError:
            # print(f"\033[91m{data.hex()}\033[0m")
            continue

    eol = bytes.fromhex("FFFFFFFF")
    PRESSURE_OFFSET = (
        70000  # So that minimum pressure is 70000 pa and max is 70000 + 2^16
    )

    cnt = 0
    while done_flag.value == 0:
        # There are 80 bytes
        # First 72 bytes are readings, next 4 bytes are time, last 4 bytes are eol
        loop_start_time = time.time()
        data = ser.read_until(eol)

        # clear_screen()
        data_numbers = []
        try:
            pressure_bytes = data[:72]
            time_bytes = data[72:76]
            newline_bytes = data[76:]

            time_int = int(time_bytes.hex(), 16)
            # print("Time: ", time_int)

            n_sensors = 36
            for i in range(n_sensors):
                pressure_hex = pressure_bytes[i * 2 : i * 2 + 2]
                pressure_int = int(pressure_hex.hex(), 16)
                pressure_int += PRESSURE_OFFSET
                data_numbers.append(pressure_int)
            cnt += 1
            data_queue.put(data_numbers + [time_int])
        except Exception as e:
            print(e)
            print(f"\033[91m{data.hex()}\033[0m")
            print(f"\033[ERROR\033[0m")
            # return
            continue

        if len(data) != 80:
            print(f"\033[91m|\n|\nERROR\n|\n|\n\033[0m")


    total_time = time.time() - total_time_start

    print(f"Hz: {cnt/total_time}")

    ser.close()

def get_color_code(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def get_line_string(line, cm_array, min_pressure=102000, max_pressure = 130000):
    # min_pressure = 70000
    # max_pressure = 70000 + 2**16
    # return " ".join([f"{val:.1f}" for val in line]) + "\n"

    str = []
    str.append("\033[48;2;171;171;173m")
    for i, val in enumerate(line):
        # val -= 100000
        color_idx = int(100 * (val - min_pressure) / (max_pressure - min_pressure))
        color_idx = min(max(color_idx, 0), 99)
        color_rgb = cm_array[color_idx]
        color_ansi = get_color_code(*color_rgb)
        # str.append(f"{color_ansi}{val:.1f}\033[38;2;0m" + " ")
        str.append(f"{color_ansi}{val:06} ")
    str = "".join(str) + "\n"

    # print(cm_array[0])
    # color = get_color_code(255, 100, 200j
    # print(f"{color}{line}\033[38;2;0]")
    return str


def plotting_target(done_flag: multiprocessing.Value, data_queue: multiprocessing.Queue, cm_array: np.ndarray):
    print("Plotting process started.")
    # https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    sensors_per_line = 12

    i = 0
    cnt = 0
    start_time = time.time()
    hz = 0
    old_max = 0
    was_max = 0
    prev_data_arr = np.zeros(3 * sensors_per_line)

    n_diff_history = 20
    diff_queue = deque()
    for _ in range(n_diff_history):
        diff_queue.append(np.zeros(3 * sensors_per_line))

    while done_flag.value == 0:

        data = data_queue.get()
        line1 = data[:sensors_per_line]
        line2 = data[sensors_per_line:2 * sensors_per_line]
        line3 = data[2 * sensors_per_line:3 * sensors_per_line]

        min_pressure = 100000
        max_pressure = 130000
        line1_str = get_line_string(line1, cm_array, min_pressure, max_pressure)
        line2_str = get_line_string(line2, cm_array, min_pressure, max_pressure)
        line3_str = get_line_string(line3, cm_array, min_pressure, max_pressure)

        clear_screen()

        data_arr = np.array(data[:3 * sensors_per_line])

        diff = data_arr - diff_queue.popleft()
        diff_queue.append(data_arr)
        
        # diff = data_arr - prev_data_arr
        # prev_data_arr = data_arr

        min_diff_color = -100
        max_diff_color = 100
        diff_str1 = get_line_string(diff[:sensors_per_line], cm_array, min_diff_color, max_diff_color)
        diff_str2 = get_line_string(diff[sensors_per_line:2 * sensors_per_line], cm_array, min_diff_color, max_diff_color)
        diff_str3 = get_line_string(diff[2 * sensors_per_line:3 * sensors_per_line], cm_array, min_diff_color, max_diff_color)


        if max(diff) > 5:
            was_max = 50

        if was_max:
            print("\033[38;2;255;0;0mCONTACT!\033[38;0m")
            was_max -= 1
        else:
            print("")


        # print(" ".join(line1))
        # print(" ".join(line2))
        # print(" ".join(line3))
        print(line1_str, end="")
        print(line2_str, end="")
        print(line3_str, end="")
        print(f"\033[38;2;0;0;0m{data[-1]} us")
        print(f"Hz: {hz}")
        print(diff_str1, end="")
        print(diff_str2, end="")
        print(diff_str3, end="")

        print(f"\033[0m", end="")  # Reset color

        cnt += 1
        if i == 300:
            hz = cnt/(time.time() - start_time)
            cnt = 0
            i = 0
            start_time = time.time()


        i += 1

def main():
    sample_time = 1/100
    # rnn_model_fname = "FA7"  # Has a different architecture...

    # Both E9 and E10 work
    rnn_model_fname = "E9"
    # rnn_model_fname = "E10"

    print(f"Starting with {rnn_model_fname}")

    colormap = cm.get_cmap("viridis")
    cm_array = colormap(np.linspace(0, 1, 100))
    cm_array = (cm_array[:, :3] * 255).astype(np.uint8)

    done_flag = multiprocessing.Value('i', 0)
    data_queue = multiprocessing.Queue()
    plotting_queue = multiprocessing.Queue()
    reader_process = multiprocessing.Process(target=serial_reader, args=(done_flag, data_queue, sample_time))

    plotting_process = multiprocessing.Process(target=plotting_target, args=(done_flag, data_queue, cm_array))

    reader_process.start()
    plotting_process.start()

    # Instantiate and start the plotter
    end_time = time.time() + 10000
    while True:
        # data = data_queue.get()
        # print(data)

        if time.time() > end_time:
            break

    done_flag.value = 1

    plotting_process.join()
    reader_process.join()
    print("Serial port closed.")

if __name__ == "__main__":
    main()
