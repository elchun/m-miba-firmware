import multiprocessing
import serial
import time
from matplotlib import cm
import numpy as np

def clear_screen():
    print("\033[H\033[J", end='')


def serial_reader(done_flag: multiprocessing.Value, data_queue: multiprocessing.Queue, sample_time: float = 1/100):
    ser = serial.Serial(
        # port='/dev/ttyACM0',
        # port='/dev/tty.usbmodem1303',
        # port='/dev/tty.usbmodem11103',
        port='/dev/tty.usbmodem1303',
        # port='/dev/tty.usbmodem212103',
        # baudrate=115200,
        baudrate=921600,
        timeout=1
    )

    if ser.isOpen():
        print(f"Serial port {ser.port} is open.")
    else:
        print(f"Failed to open serial port {ser.port}.")
        exit()

    total_time_start = time.time()

    start_time = time.perf_counter()
    cnt = 0
    while done_flag.value == 0:
        try:
            data = ser.readline().decode().strip()
        except UnicodeDecodeError:
            continue

        # if "Pressure" not in data:
        #     continue
        if time.perf_counter() - start_time > sample_time:
            cnt += 1

            try:
                # print(data)
                data_numbers = data.split(",")
                data_numbers[-1] = data_numbers[-1].split(" ")[0]
                data_numbers = [int(x) for x in data_numbers]
                # print(data_numbers)
                data_queue.put(data_numbers)
            except:
                print("Error")
                continue
            start_time += sample_time

    total_time = time.time() - total_time_start

    print(f"Hz: {cnt/total_time}")

    ser.close()

# def display_data(raw_data, Fxyzn, theta, phi, contact_prob):
#     clear_screen()
#     print("[Sensor visualization]")
#     print("Raw data")
#     print_bar_chart(raw_data)
#     print()
#     print("Predicted data")
#     print_predicted_data(Fxyzn, theta, phi, contact_prob)

def get_color_code(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def get_line_string(line, cm_array):
    min_pressure = 0
    max_pressure = 40000

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
        str.append(f"{color_ansi}{val:05} ")
    str = "".join(str) + "\n"

    # print(cm_array[0])
    # color = get_color_code(255, 100, 200j
    # print(f"{color}{line}\033[38;2;0]")
    return str


def plotting_target(done_flag: multiprocessing.Value, data_queue: multiprocessing.Queue, cm_array: np.ndarray):
    print("Plotting process started.")
    # https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    sensors_per_line = 12
    while done_flag.value == 0:
        data = data_queue.get()
        line1 = data[:sensors_per_line]
        line2 = data[sensors_per_line:2 * sensors_per_line]
        line3 = data[2 * sensors_per_line:3 * sensors_per_line]
        line1_str = get_line_string(line1, cm_array)
        line2_str = get_line_string(line2, cm_array)
        line3_str = get_line_string(line3, cm_array)
        clear_screen()
        # print(" ".join(line1))
        # print(" ".join(line2))
        # print(" ".join(line3))
        print(line1_str, end="")
        print(line2_str, end="")
        print(line3_str, end="")
        print(f"\033[38;2;0;0;0m{data[-1]} ms")
        print(f"\033[0m", end="")  # Reset color

        # print(f"{get_color_code(255, 60, 255)}{line1}\033[38;5;7m")

        # print(line2)
        # print(line3)

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
