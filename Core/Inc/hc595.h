/*
 * hc595.h
 *
 *  Created on: Nov 6, 2024
 *      Author: xinzhou
 */

#include "stm32h7xx_hal.h"

#ifndef SRC_HC595_H_
#define SRC_HC595_H_



class hc595 {
private:
	GPIO_TypeDef* _SER_PORT;
	GPIO_TypeDef* _SRCLK_PORT;
	GPIO_TypeDef* _RCLK_PORT;
	GPIO_TypeDef* _SRCLR_PORT;

	uint16_t _SER_PIN;
	uint16_t _SRCLK_PIN;
	uint16_t _RCLK_PIN;
	uint16_t _SRCLR_PIN;

	void _step();
	void (*_delay_us)(uint32_t, void*);

public:
	hc595(GPIO_TypeDef*, uint16_t, GPIO_TypeDef*, uint16_t, GPIO_TypeDef*, uint16_t, GPIO_TypeDef*, uint16_t);
	void shift(uint32_t, uint8_t);
	void clear();
	void write();
};

#endif /* SRC_HC595_H_ */
