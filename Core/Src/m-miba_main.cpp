#include "m-miba_main.h"

#include <cstdint>
#include <vector>

#include "SensorData.h"
#include "bmp5.h"
#include "hc595.h"
#include "m-miba_funcs.h"
#include "printing.h"
#include "tim.h"

volatile bool INTERRUPT_FLAG = 0;

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  if (htim->Instance == TIM5) {
    INTERRUPT_FLAG = 1;
    //
  }
}

/**
 * @brief  The application entry point.
 *
 * The data format is:
 * 80 bytes
 * [--- pressure (72 bytes) ---][--- time (4 bytes) ---][--- eol (4 bytes) ---]
 * pressure:
 * 		Every two values is a uint16 pressure reading
 * 		If p is in pascals, the transmitted value is
 * 		p - P_MIN where P_MIN = 70000pa.  This ensures
 * 		that we can represent all values with uint16
 * time:
 *  	This is a single uint32 value for the sample loop time
 * eol:
 *  	0xFFFFFFFF
 *
 */
int mmiba_main(void) {
  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_RESET);


  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // Enable DWT access
  DWT->CYCCNT = 0;                                 // Reset the cycle counter
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;             // Enable the cycle counter

  // start the timer for delay_us
  HAL_TIM_Base_Start(&htim3);
  // microsecond timer for timing code execution
  HAL_TIM_Base_Start(&htim4);

  HAL_GPIO_WritePin(SRCLK_GPIO_Port, SRCLK_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(RCLK_GPIO_Port, RCLK_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(SER1_GPIO_Port, SER1_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(SRCLR_GPIO_Port, SRCLR_Pin, GPIO_PIN_RESET);

  HAL_Delay(10);
  shiftRegister =
      new hc595(SRCLK_GPIO_Port, SRCLK_Pin, RCLK_GPIO_Port, RCLK_Pin,
                SER1_GPIO_Port, SER1_Pin, SRCLR_GPIO_Port, SRCLR_Pin);
  shiftRegister->clear();

  uint8_t num_sensors = 36;
  //
  std::vector<SensorData> sensorList;

  for (int i = 0; i < num_sensors; ++i) {
	HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    sensorList.emplace_back(i + 1);
    sensorList[i].Initialize();
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    HAL_Delay(50);
  }

  uint32_t elapsedTime = 0;
  uint16_t sample_data;

  uint8_t data_buffer[num_sensors * 2];
  uint8_t eol[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  uint8_t time_buffer[4];

  // Interrupt init before setup
  HAL_TIM_Base_Start_IT(&htim5);

  printf("DATA_BEGIN\n\r");
  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);

  while (1) {
    if (INTERRUPT_FLAG) {
      INTERRUPT_FLAG = 0;

      // Sampling time is ~2000us (2030)
      // Printing is ~700us rn
      // Loop time is set in the ioc by the interrupt timer
      // period.

      // Read sensors
      for (int i = 0; i < num_sensors; ++i) {
        sensorList[i].Sample();
        sample_data = sensorList[i].raw_data[0];
        data_buffer[2 * i] = ((sample_data >> 8) & 0xFF);
        data_buffer[2 * i + 1] = (sample_data & 0xFF);
      }

      // Send data buffer
      HAL_UART_Transmit(&huart3, data_buffer, num_sensors * 2,
                        HAL_MAX_DELAY);  // This was 779

      // Calculate elapsed time
      elapsedTime = __HAL_TIM_GET_COUNTER(&htim4);
      __HAL_TIM_SET_COUNTER(&htim4, 0);

      // Pack time buffer
      time_buffer[0] = (elapsedTime >> 8 * 3) & 0xFF;
      time_buffer[1] = (elapsedTime >> 8 * 2) & 0xFF;
      time_buffer[2] = (elapsedTime >> 8 * 1) & 0xFF;
      time_buffer[3] = (elapsedTime) & 0xFF;

      // Transmit time buffer
      HAL_UART_Transmit(&huart3, time_buffer, 4, HAL_MAX_DELAY);

      // Transmit eol buffer
      HAL_UART_Transmit(&huart3, eol, 4, HAL_MAX_DELAY);  // This was 779

    }
  }
}
