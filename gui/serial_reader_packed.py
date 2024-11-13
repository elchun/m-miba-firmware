import time
import serial


def clear_screen():
    print("\033[H\033[J", end="")


def serial_reader():
    ser = serial.Serial(port="/dev/tty.usbmodem11403", baudrate=921600, timeout=1.0)

    if ser.isOpen():
        print(f"Serial port {ser.port} is open.")
    else:
        print(f"Failed to open serial port {ser.port}.")
        exit()

    # Max pressure: 31071
    eol = bytes.fromhex("FFFFFFFF")
    PRESSURE_OFFSET = (
        70000  # So that minimum pressure is 70000 pa and max is 70000 + 2^16
    )

    print("RESET DEVICE TO BEGIN")
    while True:
        data = ser.readline()
        try:
            data_str = data.decode().strip()
            print(f"\033[38;5;{51}m{data_str}\033[0m")
            if "DATA_BEGIN" in data_str:
                break
        except UnicodeDecodeError:
            print(f"\033[91m{data.hex()}\033[0m")
            continue

    while True:
        # There are 80 bytes
        # First 72 bytes are readings, next 4 bytes are time, last 4 bytes are eol
        loop_start_time = time.time()
        data = ser.read_until(eol)

        clear_screen()
        data_numbers = []
        try:
            pressure_bytes = data[:72]
            time_bytes = data[72:76]
            newline_bytes = data[76:]

            time_int = int(time_bytes.hex(), 16)
            print("Time: ", time_int)

            n_sensors = 36
            for i in range(n_sensors):
                pressure_hex = pressure_bytes[i * 2 : i * 2 + 2]
                pressure_int = int(pressure_hex.hex(), 16)
                pressure_int += PRESSURE_OFFSET
                data_numbers.append(pressure_int)
        except Exception as e:
            print(e)
            print(f"\033[91m{data.hex()}\033[0m")
            print(f"\033[91mERRORRRRRRRRRRRRRREHELUHLIUHLIHLIUHLHL\033[0m")
            # return
            continue

        print(" ".join([str(x) for x in data_numbers]))

        if len(data) != 80:
            print(f"\033[91m|\n|\nERROR\n|\n|\n\033[0m")

        loop_end_time = time.time()

        print(f"Loop time: \033[92m{loop_end_time - loop_start_time}\033[0m")


if __name__ == "__main__":
    serial_reader()
