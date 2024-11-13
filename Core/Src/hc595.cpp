/*
 * hc595.cpp
 *
 *  Created on: Nov 6, 2024
 *      Author: xinzhou
 */

#include "hc595.h"
#include "m-miba_funcs.h"
#include "main.h"

hc595::hc595(
		GPIO_TypeDef* srclkPort, uint16_t srclkPin,
		GPIO_TypeDef* rclkPort, uint16_t rclkPin,
		GPIO_TypeDef* serPort, uint16_t serPin,
		GPIO_TypeDef* srclrPort, uint16_t srclrPin
		){

	this->_SRCLK_PORT = srclkPort;
	this->_SRCLK_PIN = srclkPin;
	this->_RCLK_PORT = rclkPort;
	this->_RCLK_PIN = rclkPin;
	this->_SER_PORT = serPort;
	this->_SER_PIN = serPin;
	this->_SRCLR_PORT = srclrPort;
	this->_SRCLR_PIN = srclrPin;
	this->_delay_us = &bmp_delay_us;

	HAL_GPIO_WritePin(this->_SRCLK_PORT, this->_SRCLK_PIN, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(this->_RCLK_PORT, this->_RCLK_PIN, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(this->_SER_PORT, this->_SER_PIN, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(this->_SRCLR_PORT, this->_SRCLR_PIN, GPIO_PIN_SET);

	this->clear();
}

void hc595::shift(uint32_t data, uint8_t size) {

	if( size > 0 ) {
		for(int i=0; i<size; i++) {
			HAL_GPIO_WritePin(
					this->_SER_PORT,
					this->_SER_PIN,
					(0x01 & (data >> i)) ? GPIO_PIN_SET : GPIO_PIN_RESET);

			this->_step();
		}

  }
}

void hc595::_step() {
	HAL_GPIO_WritePin(this->_SRCLK_PORT, this->_SRCLK_PIN, GPIO_PIN_SET);
//	this->_delay_us(1, nullptr);
	HAL_GPIO_WritePin(this->_SRCLK_PORT, this->_SRCLK_PIN, GPIO_PIN_RESET);
//	this->_delay_us(1, nullptr);
}

void hc595::write() {
	HAL_GPIO_WritePin(this->_RCLK_PORT, this->_RCLK_PIN, GPIO_PIN_SET);
//	this->_delay_us(1, nullptr);
	HAL_GPIO_WritePin(this->_RCLK_PORT, this->_RCLK_PIN, GPIO_PIN_RESET);
//	this->_delay_us(1, nullptr);
}

void hc595::clear(){
	HAL_GPIO_WritePin(this->_SRCLR_PORT, this->_SRCLR_PIN, GPIO_PIN_RESET);
//	this->_delay_us(1, nullptr);
	HAL_GPIO_WritePin(this->_SRCLR_PORT, this->_SRCLR_PIN, GPIO_PIN_SET);
//	this->_delay_us(1, nullptr);
	write();
}



