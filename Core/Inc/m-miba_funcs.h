#ifndef BMP5_FUNCS_H
#define BMP5_FUNCS_H

#include <cstdint>
#include <cstring>
#include <cstdio>
#include "main.h"
#include "hc595.h"
#include "tim.h"


// hardware prototypes
extern SPI_HandleTypeDef hspi1;
extern hc595* shiftRegister;

// function prototypes
void writeLow(uint8_t pin);
void writeHigh();

int8_t bmp_spi1_read(uint8_t cspin, uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr_);
int8_t bmp_spi1_write(uint8_t cspin, uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr_);


void bmp_delay_us(uint32_t usec, void *intf_ptr);

#endif
