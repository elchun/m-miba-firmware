#include "SensorData.h"
#include "main.h"

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

    HAL_Delay(1);
//    rslt = bmp5_soft_reset(dev);
//    printf("bmp5 soft reset result: %d\n \r", rslt);
//    HAL_Delay(1000);


//    while(1){
//    uint8_t dev_id;
//    rslt = bmp_spi1_read(s.dev_id, 0x81, &dev_id, 1, &s);
//    printf("Device ID: 0x%02X, read result: %d\n \r", dev_id, rslt);
//    }
//    enum bmp5_powermode power_mode;
//    rslt = bmp5_get_power_mode(&power_mode, dev);
//    printf("Power mode after reset: %d, result: %d\n \r", power_mode, rslt);

//    uint8_t nvm_status;

//    rslt = get_nvm_status(&nvm_status, dev);
//    printf("nvm_status: %d, result: %d\n \r", nvm_status, rslt);


    rslt = bmp5_init(dev);
//    printf("bmp5_init result: %d\n \r", rslt);
    HAL_Delay(5);
    if (rslt == BMP5_OK) {
        enum bmp5_powermode check_pwr;
        rslt = bmp5_get_power_mode(&check_pwr, dev);
//        printf("* Power mode before = 0x%x *\n", check_pwr);

        rslt = bmp5_set_osr_odr_press_config(&osr_config, dev);

        if (rslt == BMP5_OK) {
            rslt = bmp5_set_power_mode(BMP5_POWERMODE_CONTINOUS, dev);
            HAL_Delay(1);  // Delay 1 ms
            bmp5_get_power_mode(&check_pwr, dev);
//            printf("* Power mode after = 0x%x *\n", check_pwr);

            HAL_Delay(10);  // Delay 10 ms
            if (rslt != BMP5_OK) {
                printf("Error on power mode setup.\n\r");
            }else{
            	printf("done.\n\r");
            }
        } else {
            printf("Error on settings setup.\n\r");
        }
    } else {
        printf("Error on initialization.\n \r");
    }

}

void SensorData::Sample() {
    bmp5_get_sensor_data(&data, &osr_config, &s);

    raw_data[0] = int(data.pressure) - 100000;
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
