import re
import torch
import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import multiprocessing
import queue
import json
from vector_plotter import RealTime3DVectorPlotter

from scipy.spatial.transform import Rotation as R

# from experiment_factory import get_experiment_dir
from models import *
# from datasets import *

def get_experiment_dir(experiment_name):
    return f"experiments/{experiment_name}"

def clear_screen():
    print("\033[H\033[J", end='')

def serial_reader(done_flag: multiprocessing.Value, data_queue: multiprocessing.Queue, sample_time: float = 1/100):
    ser = serial.Serial(
        # port='/dev/ttyACM0',
        #port='/dev/tty.usbmodem1303',
        port='/dev/tty.usbmodem212103',
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

        if "Pressure" not in data:
            continue

        if time.perf_counter() - start_time > sample_time:
            cnt += 1

            data_numbers = [float(num.strip()) for num in data.split(":")[1].split(" ") if num.strip()]
            data_queue.put(data_numbers)
            start_time += sample_time

    total_time = time.time() - total_time_start

    print(f"Hz: {cnt/total_time}")

    ser.close()


def plot_data(queue):
    max_plot_size = 100
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], lw=2)[0] for _ in range(8)]
    ax.set_ylim(-50000, 50000)
    ax.set_xlim(0, max_plot_size + 10)
    ax.grid()
    ax.legend([f"s{i}" for i in range(8)])
    xdata = [[0] for _ in range(8)]
    ydata = [[0] for _ in range(8)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        print(time.time())
        while not queue.empty():
            data = queue.get()
            # (raw_data, Fxyzn, weighted_theta_deg, weighted_phi_deg, contact_prob) = queue.get()

            # print(raw_data)
            # print("Fn: ", Fxyzn[..., -1], "theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "contact", contact_prob)

            for i in range(8):
                ydata[i].append(data[i])
                xdata[i].append(xdata[i][-1] + 1)

                if len(xdata[i]) > max_plot_size:
                    xdata[i].pop(0)
                    ydata[i].pop(0)

            ax.set_xlim(xdata[0][0], xdata[0][0] + max_plot_size + 10)

        for i, line in enumerate(lines):
            line.set_data(xdata[i], ydata[i])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=lambda: iter(int, 1), init_func=init, blit=True, cache_frame_data=False)
    plt.show()


def load_model(experiment_name):
    # Use cpu
    device = "cpu"

    # Load the model
    experiment_dir = get_experiment_dir(experiment_name)
    weight_dir = f"{experiment_dir}/weights"
    with open(f"{experiment_dir}/experiment_info.json", "r") as f:
        experiment_info = json.load(f)

    try:
        with open(f"{experiment_dir}/mean_X.json") as f:
            mean_X = json.load(f)

        with open(f"{experiment_dir}/std_dev_X.json") as f:
            std_dev_X = json.load(f)

        print(mean_X)
        print(std_dev_X)
    except FileNotFoundError as e:
        print(e, "Run real_time_deploy.py with the same experiment name to generate mean and std_dev files.")
        raise e


    std_dev_X = torch.tensor(std_dev_X).float()
    mean_X = torch.tensor(mean_X).float()

    model = eval(experiment_info["model_name"])(**experiment_info["model_kwargs"]).to(device)
    model.load_state_dict(torch.load(f"{weight_dir}/best.pt", map_location="cpu"))
    return model, std_dev_X, mean_X


def run_rnn(data_queue, model, std_dev_X, mean_X, done_flag):

    model.eval()
    h = torch.randn(model.rnn.num_layers, model.rnn.hidden_size, device="cpu")
    while done_flag.value == 0:
        if not data_queue.empty():
            data = data_queue.get()
            data = torch.tensor(data).unsqueeze(0).float()
            data = (data - mean_X) / std_dev_X

            h, _ = model.rnn(data, h)
            y_pred = model.fc(h).detach()

            Fxyzn = y_pred[..., :4]
            theta_rad = y_pred[..., 4]
            phi_rad = y_pred[..., 5]
            contact_flag = y_pred[..., -1]

            theta_deg = theta_rad * 180 / np.pi
            phi_deg = phi_rad * 180 / np.pi
            contact_prob = torch.sigmoid(contact_flag)

            print("F: ", Fxyzn[..., -1], "theta: ", theta_deg, "phi: ", phi_deg, "contact", contact_prob)

def run_binned_rnn(data_queue, plotting_queue, model, std_dev_X, mean_X, done_flag):
    model.eval()
    h = torch.randn(model.rnn.num_layers, model.rnn.hidden_size, device="cpu")

    theta_angles = [model.theta_idx_to_angle(i) for i in range(model.n_theta_bins)]
    phi_angles = [model.phi_idx_to_angle(i) for i in range(model.n_phi_bins)]

    theta_angles = torch.tensor(theta_angles).unsqueeze(0).float()
    phi_angles = torch.tensor(phi_angles).unsqueeze(0).float()
    while done_flag.value == 0:
        if not data_queue.empty():
            raw_data = data_queue.get()
            raw_data = torch.tensor(raw_data).unsqueeze(0).float()  # L=1, H_in = 8
            data = (raw_data - mean_X) / std_dev_X

            h, _ = model.rnn(data, h)
            y_pred = model.fc(h)

            y_pred = y_pred.detach()
            h = h.detach()

            Fxyzn = y_pred[..., :4]
            theta_bins = y_pred[..., 4:4 + model.n_theta_bins]
            phi_bins = y_pred[..., 4 + model.n_theta_bins:4 + model.n_theta_bins + model.n_phi_bins]
            contact_flag = y_pred[..., -1:]

            theta_probs = torch.softmax(theta_bins, dim=-1)
            phi_probs = torch.softmax(phi_bins, dim=-1)

            theta_idx = torch.argmax(theta_probs, dim=-1)
            phi_idx = torch.argmax(phi_probs, dim=-1)

            theta = model.theta_idx_to_angle(theta_idx)
            phi = model.phi_idx_to_angle(phi_idx)

            weighted_theta = torch.sum(theta_probs * theta_angles)
            weighted_phi = torch.sum(phi_probs * phi_angles)

            weighted_theta_deg = weighted_theta * 180 / np.pi
            weighted_phi_deg = weighted_phi * 180 / np.pi

            theta_deg = theta * 180 / np.pi
            phi_deg = phi * 180 / np.pi

            contact_prob = torch.sigmoid(contact_flag)

            display_data(raw_data, Fxyzn, weighted_theta_deg, weighted_phi_deg, contact_prob)



            xyz, uvw, color = prepare_data_for_queue(Fxyzn, weighted_theta_deg, weighted_phi_deg, contact_prob)
            # print(xyz, uvw)
            plotting_queue.put((*xyz, *uvw, color))

            # print("Theta probs: ", np.round(theta_probs, 3))
            # print("theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "theta: ", theta_deg, "phi: ", phi_deg, "contact", contact_prob)

            # plotting_queue.put((raw_data, Fxyzn, weighted_theta_deg, weighted_phi_deg, contact_prob))

            # Print the bar chart
            # print(Fxyzn, theta_probs, phi_probs, contact_flag)

            # time.sleep(0.1)


def prepare_data_for_queue(Fxyzn, theta, phi, contact_prob):
    CONTACT_THRESH = 0.2

    initial_start = np.array([0, 0, 1])
    R_theta  = R.from_euler('x', -theta, degrees=True)
    R_phi = R.from_euler('y', phi, degrees=True)

    R_total = R_phi * R_theta
    start_vector = R_total.apply(initial_start)

    end_vector = start_vector + Fxyzn[..., :3].squeeze().numpy()
    return start_vector, end_vector, 'g' if contact_prob > CONTACT_THRESH else 'r'


def display_data(raw_data, Fxyzn, theta, phi, contact_prob):
    clear_screen()
    print("[Sensor visualization]")
    print("Raw data")
    print_bar_chart(raw_data)
    print()
    print("Predicted data")
    print_predicted_data(Fxyzn, theta, phi, contact_prob)
    # print("Fn: ", Fxyzn[..., -1], "theta: ", theta, "phi: ", phi, "contact", contact_prob)

def print_bar_chart(raw_data: np.ndarray):
    MAX_DATA = 70000
    PROW_WIDTH = 100
    MAX_NROW = 30
    N_LJUST = 30
    data_normalized = raw_data.flatten() / MAX_DATA

    red = "\033[91m"
    green = "\033[92m"
    reset = "\033[0m"

    lines = []
    for i in range(len(data_normalized)):
        pbar_cnt = max(int(data_normalized[i] * PROW_WIDTH), 0)
        pspace_cnt = PROW_WIDTH - pbar_cnt

        nbar_cnt = -min(int(data_normalized[i] * PROW_WIDTH), 0)
        nspace_cnt = MAX_NROW - nbar_cnt

        # print(data_normalized[i] * PROW_WIDTH)

        raw_sensor_value = int(raw_data[0, i])
        line = f"s{i + 1}: {str(raw_sensor_value).rjust(6, ' ')}".ljust(N_LJUST) + f"[{red}{' '*nspace_cnt}{'#'*nbar_cnt}{reset}|{green}{'#' * pbar_cnt}{' ' * pspace_cnt}{reset}]"
        lines.append(line)


    # Remove ANSI escape sequences using a regular expression
    # Actual black magic
    visible_line = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line)

    # Get the length of the string without ANSI escape codes
    line_length = len(visible_line)

    # print("-" * (line_length))
    print("\n".join(lines))
    # print("-" * (line_length))


def print_predicted_data(Fxyxn, theta, phi, contact_prob):
    red = "\033[91m"
    green = "\033[92m"
    orange = "\033[93m"
    reset = "\033[0m"
    N_LJUST = 30


    MAX_FORCE = 10
    FORCE_BAR_WIDTH = 10

    lines = []
    # print Fx

    force_labels = ["Fx", "Fy", "Fz", "Fn"]

    for i in range(3):
        bar_cnt = int(Fxyxn[..., i].item() / MAX_FORCE * FORCE_BAR_WIDTH)

        pbar_cnt = max(bar_cnt, 0)
        nbar_cnt = -min(bar_cnt, 0)

        p_space_cnt = FORCE_BAR_WIDTH - pbar_cnt
        n_space_cnt = FORCE_BAR_WIDTH - nbar_cnt

        force_string = f"{Fxyxn[..., i].item():.2f}N".rjust(6, ' ')
        line = f"{force_labels[i]}: {force_string}".ljust(N_LJUST) + f"[{red}{' ' * n_space_cnt}{'#'*nbar_cnt}{reset}|{green}{bar_cnt * '#'}{reset}{' ' * p_space_cnt}]"
        lines.append(line)

    # print Fn
    # NORMAL_BAR_WIDTH = FORCE_BAR_WIDTH * 2 + 1
    NORMAL_BAR_WIDTH = 21
    MAX_NORMAL_FORCE = 21
    bar_cnt = -int(Fxyxn[..., -1].item() / MAX_NORMAL_FORCE * NORMAL_BAR_WIDTH)
    force_string = f"{-1 * Fxyxn[..., -1].item():.2f}N".rjust(6, ' ')
    line = f"{force_labels[-1]}: {force_string}".ljust(N_LJUST) + f"[{green}{'#'*bar_cnt}{reset}{' ' * ( NORMAL_BAR_WIDTH - bar_cnt)}]"
    lines.append(line)

    # Plot theta
    BIN_WIDTH = 5
    labels = ["Theta:", "Phi:"]
    min_ranges = [-45, -135]
    max_ranges = [45, 45]
    angles = [theta, phi]
    start_paddings = [(min_ranges[i] - min(min_ranges)) // BIN_WIDTH for i in range(2)]
    for i in range(2):
        MIN_ANGLE = min_ranges[i]
        MAX_ANGLE = max_ranges[i]
        BAR_WIDTH = (MAX_ANGLE - MIN_ANGLE) // BIN_WIDTH
        start_padding = start_paddings[i] * ' '

        angle_str = [' ' for _ in range(BAR_WIDTH)]
        bar_idx = int((angles[i] - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) * BAR_WIDTH)
        angle_str[bar_idx] = '|'
        angle_str = ''.join(angle_str)

        gt_angle_str = f"{angles[i]:.3f}°".rjust(7, ' ')

        line = f"{labels[i].ljust(6, ' ')} {gt_angle_str}".ljust(N_LJUST) + f"{start_padding}[{orange}{angle_str}{reset}]"
        lines.append(line)


    # Contact flag
    N_CONTACT_BINS = 10
    CONTACT_THRESH = 0.2

    N_RED_BINS = int(N_CONTACT_BINS * CONTACT_THRESH)

    contact_bar = int(contact_prob.item() * N_CONTACT_BINS)
    red_contact_bins = min(contact_bar, N_RED_BINS)

    green_contact_bins = contact_bar - red_contact_bins

    contact_str = f"[{red}{'#' * red_contact_bins}{green}{'#' * green_contact_bins}{reset}{reset}{ ' ' * (N_CONTACT_BINS - contact_bar)}]"

    gt_contact_str = f"{contact_prob.item():.3f}".rjust(5, ' ')

    line = f"Contact Flag: {gt_contact_str}".ljust(N_LJUST) + f"{contact_str}"

    lines.append(line)




    print("\n".join(lines))






def run_binned_lstm(data_queue, model, std_dev_X, mean_X, done_flag):
    model.eval()
    h = torch.randn(model.rnn.num_layers, model.rnn.hidden_size, device="cpu")
    c = torch.randn(model.rnn.num_layers, model.rnn.hidden_size, device="cpu")

    theta_angles = [model.theta_idx_to_angle(i) for i in range(model.n_theta_bins)]
    phi_angles = [model.phi_idx_to_angle(i) for i in range(model.n_phi_bins)]

    theta_angles = torch.tensor(theta_angles).unsqueeze(0).float()
    phi_angles = torch.tensor(phi_angles).unsqueeze(0).float()
    while done_flag.value == 0:
        if not data_queue.empty():
            raw_data = data_queue.get()
            raw_data = torch.tensor(raw_data).unsqueeze(0).float()  # L=1, H_in = 8
            data = (raw_data - mean_X) / std_dev_X

            h, (_, c) = model.rnn(data, (h, c))
            print(c)
            y_pred = model.fc(h)

            y_pred = y_pred.detach()
            h = h.detach()

            Fxyzn = y_pred[..., :4]
            theta_bins = y_pred[..., 4:4 + model.n_theta_bins]
            phi_bins = y_pred[..., 4 + model.n_theta_bins:4 + model.n_theta_bins + model.n_phi_bins]
            contact_flag = y_pred[..., -1:]

            theta_probs = torch.softmax(theta_bins, dim=-1)
            phi_probs = torch.softmax(phi_bins, dim=-1)

            theta_idx = torch.argmax(theta_probs, dim=-1)
            phi_idx = torch.argmax(phi_probs, dim=-1)

            theta = model.theta_idx_to_angle(theta_idx)
            phi = model.phi_idx_to_angle(phi_idx)

            weighted_theta = torch.sum(theta_probs * theta_angles)
            weighted_phi = torch.sum(phi_probs * phi_angles)

            weighted_theta_deg = weighted_theta * 180 / np.pi
            weighted_phi_deg = weighted_phi * 180 / np.pi

            theta_deg = theta * 180 / np.pi
            phi_deg = phi * 180 / np.pi

            contact_prob = torch.sigmoid(contact_flag)

            # print("Theta probs: ", np.round(theta_probs, 3))
            # print("theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "theta: ", theta_deg, "phi: ", phi_deg, "contact", contact_prob)

            print(raw_data)
            print("Fn: ", Fxyzn[..., -1], "theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "contact", contact_prob)
            # print(Fxyzn, theta_probs, phi_probs, contact_flag)


def run_mlp(queue, model, std_dev_X, mean_X, done_flag):

    model.eval()
    while done_flag.value == 0:
        if not queue.empty():
            data = queue.get()
            data = torch.tensor(data).unsqueeze(0).float()
            data = (data - mean_X) / std_dev_X

            y_pred = model.std_forward(data).detach()
            print(y_pred)

            Fx = y_pred[..., 0]
            Fy = y_pred[..., 1]
            Fz = y_pred[..., 2]
            Fn = y_pred[..., 3]
            theta = y_pred[..., 4]
            phi = y_pred[..., 5]
            contact_prob = y_pred[..., 6]

            theta_deg = theta * 180 / np.pi
            phi_deg = phi * 180 / np.pi

            print("theta: ", theta_deg.item(), "phi: ", phi_deg.item(), "contact", contact_prob.item())


def run_mlp_and_rnn(data_queue, mlp_model, rnn_model, std_dev_X, mean_X, done_flag):
    mlp_model.eval()
    rnn_model.eval()


    h = torch.randn(rnn_model.rnn.num_layers, rnn_model.rnn.hidden_size, device="cpu")

    theta_angles = [rnn_model.theta_idx_to_angle(i) for i in range(rnn_model.n_theta_bins)]
    phi_angles = [rnn_model.phi_idx_to_angle(i) for i in range(rnn_model.n_phi_bins)]

    theta_angles = torch.tensor(theta_angles).unsqueeze(0).float()
    phi_angles = torch.tensor(phi_angles).unsqueeze(0).float()
    while done_flag.value == 0:
        if not data_queue.empty():
            raw_data = data_queue.get()
            raw_data = torch.tensor(raw_data).unsqueeze(0).float()  # L=1, H_in = 8
            data = (raw_data - mean_X) / std_dev_X

            h, _ = rnn_model.rnn(data, h)
            y_pred_rnn = rnn_model.fc(h)
            y_pred_mlp = mlp_model.std_forward(data)

            y_pred_rnn = y_pred_rnn.detach()
            h = h.detach()

            Fxyzn = y_pred_rnn[..., :4]
            theta_bins = y_pred_rnn[..., 4:4 + rnn_model.n_theta_bins]
            phi_bins = y_pred_rnn[..., 4 + rnn_model.n_theta_bins:4 + rnn_model.n_theta_bins + rnn_model.n_phi_bins]
            contact_flag = y_pred_rnn[..., -1:]

            theta_probs = torch.softmax(theta_bins, dim=-1)
            phi_probs = torch.softmax(phi_bins, dim=-1)

            theta_idx = torch.argmax(theta_probs, dim=-1)
            phi_idx = torch.argmax(phi_probs, dim=-1)

            theta = rnn_model.theta_idx_to_angle(theta_idx)
            phi = rnn_model.phi_idx_to_angle(phi_idx)

            weighted_theta = torch.sum(theta_probs * theta_angles)
            weighted_phi = torch.sum(phi_probs * phi_angles)

            weighted_theta_deg = weighted_theta * 180 / np.pi
            weighted_phi_deg = weighted_phi * 180 / np.pi

            theta_deg = theta * 180 / np.pi
            phi_deg = phi * 180 / np.pi

            contact_prob = torch.sigmoid(contact_flag)

            # print("Theta probs: ", np.round(theta_probs, 3))
            # print("theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "theta: ", theta_deg, "phi: ", phi_deg, "contact", contact_prob)

            print(raw_data)
            print("RNN: ", "Fn: ", Fxyzn[..., -1], "theta: ", weighted_theta_deg, "phi: ", weighted_phi_deg, "contact", contact_prob)



            theta_mlp = y_pred_mlp[..., 4]
            phi_mlp = y_pred_mlp[..., 5]
            theta_mlp_deg = theta_mlp * 180 / np.pi
            phi_mlp_deg = phi_mlp * 180 / np.pi
            print("MLP: ", "Fn:", y_pred_mlp[..., 3], "theta: ", theta_mlp_deg, "phi: ", phi_mlp_deg, "contact", y_pred_mlp[..., 6])
            # print(Fxyzn, theta_probs, phi_probs, contact_flag)


def main():
    sample_time = 1/100
    # rnn_model_fname = "FA7"  # Has a different architecture...

    # Both E9 and E10 work
    rnn_model_fname = "E9"
    # rnn_model_fname = "E10"

    print(f"Starting with {rnn_model_fname}")

    done_flag = multiprocessing.Value('i', 0)
    data_queue = multiprocessing.Queue()
    plotting_queue = multiprocessing.Queue()
    reader_process = multiprocessing.Process(target=serial_reader, args=(done_flag, data_queue, sample_time))
    # plotter_process = multiprocessing.Process(target=plot_data, args=(plotting_queue,))


    rnn_model, std_dev_X, mean_X = load_model(rnn_model_fname)
    model_process = multiprocessing.Process(target=run_binned_rnn, args=(data_queue, plotting_queue, rnn_model, std_dev_X, mean_X, done_flag))

    model_process.start()
    # plotter_process.start()
    reader_process.start()

    # Instantiate and start the plotter
    plotter = RealTime3DVectorPlotter(queue=plotting_queue, update_interval=100)
    plt.show()
    end_time = time.time() + 10000
    while True:
        # data = data_queue.get()
        # print(data)

        if time.time() > end_time:
            break

    done_flag.value = 1

    model_process.join()
    plotter_process.join()
    reader_process.join()
    print("Serial port closed.")

if __name__ == "__main__":
    main()
