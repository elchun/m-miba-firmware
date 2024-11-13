#include "m-miba_funcs.h"

hc595* shiftRegister = nullptr;


void writeLow(uint8_t pin){

	uint8_t stripNum;
	uint32_t baroNum;
	stripNum = (pin-1)/12+1;
	baroNum = (pin-1)%12+1;
	shiftRegister->shift(baroNum, stripNum*4);
	shiftRegister->write();

}

void writeHigh(){
	shiftRegister->clear();
}

// read function: |0x80 done in library, dummy byte taken care of in library
// (uint8_t cspin, uint8_t reg_addr, uint8_t *read_data, uint32_t len, void *intf_ptr_)
int8_t bmp_spi1_read(uint8_t cspin, uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr_) {
    writeLow(cspin);  // Set CS low to start the transaction

    // Send the register address to initiate the read
    HAL_StatusTypeDef status = HAL_SPI_Transmit(&hspi1, &reg_addr, 1, HAL_MAX_DELAY);
    if (status != HAL_OK) {
        writeHigh();  // Set CS high to end the transaction in case of error
        return -1;
    }

    // Read each byte individually

    for (int i = 0; i < len; i++) {
        status = HAL_SPI_Receive(&hspi1, &reg_data[i], 1, HAL_MAX_DELAY);
        if (status != HAL_OK) {
            writeHigh();  // Set CS high to end the transaction in case of error
            return -1;
        }
    }

    writeHigh();  // Set CS high to end the transaction
    return 0;     // Return 0 on success
}

int8_t bmp_spi1_write(uint8_t cspin, uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr_) {
    writeLow(cspin);  // Set CS low to start the transaction

    // Send the register address
    HAL_StatusTypeDef status = HAL_SPI_Transmit(&hspi1, &reg_addr, 1, HAL_MAX_DELAY);
    if (status != HAL_OK) {
        writeHigh();  // Set CS high to end the transaction in case of error
        printf("Write failed \n\r");
        return -1;
    }

    // Write each byte individually
    for (int i = 0; i < len; i++) {
        status = HAL_SPI_Transmit(&hspi1, (uint8_t*)&reg_data[i], 1, HAL_MAX_DELAY);
        if (status != HAL_OK) {
            writeHigh();  // Set CS high to end the transaction in case of error
            printf("Read failed \n\r");
            return -1;
        }
    }

    writeHigh();  // Set CS high to end the transaction
    return 0;     // Return 0 on success
}

// Delay function
void bmp_delay_us(uint32_t usec, void *intf_ptr) {
//    uint32_t startTick = DWT->CYCCNT;
//    uint32_t delayTicks = usec * (SystemCoreClock / 1000000);
//
//    while ((DWT->CYCCNT - startTick) < delayTicks);
	delay_us(usec);
}

//void bmp_delay_us(uint32_t usec, void *intf_ptr) {
//    HAL_Delay(usec / 1000);  // Approximate conversion to milliseconds for testing
//}
//
