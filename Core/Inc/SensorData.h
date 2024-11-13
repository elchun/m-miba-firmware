#ifndef SENSORDATA_H
#define SENSORDATA_H

#include "bmp5.h"
#include "m-miba_funcs.h"
#include "main.h"  // Include main header for HAL libraries
//#include <cstdio>
#include <stdio.h>

// hardware prototypes
extern SPI_HandleTypeDef hspi1; // Use STM32 HAL SPI handle


class SensorData {
public:
    SensorData(int sensor_number);
    void Sample();
    void Initialize();
    void Calibrate();

    int _sensor; // sensor number
    int16_t raw_data[1]; // data from pressure sensor
    int offsets[1]; // offset to subtract from raw data
    int offset_data[1]; // updated reading after subtracting offset from raw data

private:
    void config_dev(struct bmp5_dev *dev);

    uint8_t sensor_comp;

    struct bmp5_dev s; // Device structure
    struct bmp5_sensor_data data; // Sensor data structure
    struct bmp5_osr_odr_press_config osr_config = {
        .osr_t = BMP5_OVERSAMPLING_1X, // 1x oversampling for temperature
        .osr_p = BMP5_OVERSAMPLING_1X, // 1x oversampling for pressure
        .press_en = BMP5_ENABLE,       // Pressure sampling enable
        .odr = BMP5_ODR_240_HZ
    };
};

#endif
