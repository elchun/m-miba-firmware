import serial

def serial_reader():
    ser = serial.Serial(
        # port='/dev/ttyACM0',
        # port='/dev/tty.usbmodem1303',
        port='/dev/tty.usbmodem11403',
        # port='/dev/tty.usbmodem11103',
        # port='/dev/tty.usbmodem212103',
        baudrate=921600,
        timeout=1
    )

    # char is 2^8 which in hex is 2 digits
    if ser.isOpen():
        print(f"Serial port {ser.port} is open.")
    else:
        print(f"Failed to open serial port {ser.port}.")
        exit()

    # Max pressure: 31071
    while True:
        try:
            # 146 hex digits  (144 for data, 2 for newline)

            # Can do if full and if newline is at the end


            data_bytes = ser.readline()
            print(len(data_bytes))


            # print(data_bytes.decode().strip())
            # 72 bytes for data, remaing are time and newline
            data_time = data_bytes[72:]
            data_time_decoded = data_time.decode().strip()
            print(data_time_decoded)


            # print(len(ser.readline().hex()), end = " ")
            # data_hex = ser.readline().hex()
            # print(data_hex)

            # time_and_newline = data_hex[144:]


            # time = bytes(time_and_newline, "utf-8").decode().strip()
            # print(time)


            # data_arr = []
            # for h in range(0, len(data_hex), 4):
            #     reading = int(data_hex[h:h+4], 16)
            #     data_arr.append(reading)
            # # print(data_arr[0])



            # print()
            # data = ser.readline().decode().strip()
            # print(data)
            # print(len(data))
        except UnicodeDecodeError:
            print("\33[31mUnicodeDecodeError\33[0m")
            continue

if __name__ == "__main__":
    serial_reader()