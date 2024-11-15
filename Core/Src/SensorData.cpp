#include "SensorData.h"
#include "main.h"

#define BMP5_USE_FIXED_POINT

SensorData::SensorData(int sensor_number) {
    _sensor = sensor_number;
    sensor_comp = uint8_t(1);
}

void SensorData::Initialize() {
    printf("\nInitializing channel %d.\n\r", _sensor);
    s.dev_id = _sensor;  // Assign device ID
    config_dev(&s);
}

void SensorData::config_dev(struct bmp5_dev *dev) {
    int8_t rslt = BMP5_OK;

    dev->intf = BMP5_SPI_INTF;  // Use SPI interface
    dev->read = &bmp_spi1_read; // Function pointers for SPI communication
    dev->write = &bmp_spi1_write;
    dev->delay_us = &bmp_delay_us;
//    dev->delay_us = &HAL_Delay;


    // Init init
    rslt = bmp5_init(dev);
    printf("bmp5_init result: %d\n \r", rslt);

    if (rslt == BMP5_OK) {
    	printf("Init passed\n");
    } else {
    	printf("Error on initialization.\n \r");
    }


    // osr odr setup
    rslt = bmp5_set_osr_odr_press_config(&osr_config, dev);

    if (rslt == BMP5_OK) {
    	printf("Settings setup passed\n\r");
    } else {
    	printf("Error on settings setup.\n\r");
    }

    // Powermode setup
    rslt = bmp5_set_power_mode(BMP5_POWERMODE_CONTINOUS, dev);
    if (rslt == BMP5_OK) {
    	printf("Power setup passed\n\r");

    } else {
    	printf("Error on power mode setup.\n\r");
    }

}

void SensorData::Sample() {
//	bmp5_set_power_mode(BMP5_POWERMODE_CONTINOUS, &s);
//    bmp5_get_sensor_data(&data, &osr_config, &s);
    bmp5_get_sensor_data_ponly(&data, &osr_config, &s);

    // Min measured pressure is 70000
    raw_data[0] = int(data.pressure) - 70000;  // So there are no negatives...
//    offset_data[0] = raw_data[0] - offsets[0];
}

void SensorData::Calibrate() {
    printf("Calculating sensor offsets.\n");
    float temp_offsets[] = {0.0f};
    int num_samples = 10;

    for (int i = 0; i < num_samples; i++) {
        Sample();
        temp_offsets[0] += ((float)raw_data[0]) / num_samples;
        HAL_Delay(10);  // Delay 10 ms
    }

    offsets[0] = (int)temp_offsets[0];
    printf("Saved offsets: %d\n", offsets[0]);
    HAL_Delay(100);  // Delay 100 ms
}
