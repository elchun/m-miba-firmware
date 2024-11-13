#include "m-miba_main.h"
#include "printing.h"

#include "bmp5.h"
#include "m-miba_funcs.h"
#include "SensorData.h"
#include "hc595.h"
#include <cstdint>
#include <vector>

#include "tim.h"


volatile bool INTERRUPT_FLAG = 0;


void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {

	if (htim->Instance == TIM5) {
		INTERRUPT_FLAG = 1;
		//
	}
}


int mmiba_main(void) {

	printf("Hello World!\n\r");
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;   // Enable DWT access
	DWT->CYCCNT = 0;                                  // Reset the cycle counter
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;              // Enable the cycle counter

	// start the timer for delay_us
	HAL_TIM_Base_Start(&htim3);
	// microsecond timer for timing code execution
	HAL_TIM_Base_Start(&htim4);


	HAL_GPIO_WritePin(SRCLK_GPIO_Port, SRCLK_Pin, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(RCLK_GPIO_Port, RCLK_Pin, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(SER1_GPIO_Port, SER1_Pin, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(SRCLR_GPIO_Port, SRCLR_Pin, GPIO_PIN_RESET);

	HAL_Delay(10);
	shiftRegister = new hc595(SRCLK_GPIO_Port, SRCLK_Pin, RCLK_GPIO_Port, RCLK_Pin, SER1_GPIO_Port, SER1_Pin, SRCLR_GPIO_Port, SRCLR_Pin);
	shiftRegister->clear();

	uint8_t num_sensors = 36;
	//
	std::vector<SensorData> sensorList;

	for (int i = 0; i < num_sensors; ++i) {
	  sensorList.emplace_back(i+1);
	  sensorList[i].Initialize();
	  HAL_Delay(100);
	}


	uint32_t previousTime = 0;
	uint32_t currentTime = 0;
	uint32_t elapsedTime = 0;
	uint32_t elapsedTimeNoPrint = 0;
	uint32_t sample_start_time  = 0;

	uint8_t first_byte;
	uint8_t second_byte;
	uint16_t sample_data;

	// Interrupt init before setup
	HAL_TIM_Base_Start_IT(&htim5);


//	int data_arr[num_sensors];

	while (1)
	{
		if (INTERRUPT_FLAG) {
			  INTERRUPT_FLAG = 0;





		 // Sampling time is ~2ms (2030)
			// Prining is ~3ms rn

	//	  char data_str[100];
	//	  sample_start_time = __HAL_TIM_GET_COUNTER(&htim4);
		  for (int i = 0; i < num_sensors; ++i) {
			  sensorList[i].Sample();



			  printf("%d,", sensorList[i].raw_data[0]);
	//		  HAL_UART_Transmit(&huart3, (uint8_t*) &sensorList[i].raw_data[0], 1, HAL_MAX_DELAY);


//			  uint16_t data_uint16_t = (uint16_t) sensorList[i].raw_data[0];
//			  data_str += (char*) sensorList[i].raw_data[0];
//			  printf("aa");
			  sample_data = sensorList[i].raw_data[0];
			  first_byte = (uint8_t) ((sample_data >> 8) & 0xFF);  // <first_byte><second_byte> = <value in binary>
			  second_byte = (uint8_t) (sample_data & 0xFF);

//			  printf("%c%c", first_byte, second_byte);
		  }

	//	  currentTime = HAL_GetTick();
	//	  elapsedTime = currentTime - previousTime;
	//	  previousTime = currentTime;
		  elapsedTime = __HAL_TIM_GET_COUNTER(&htim4);
		  __HAL_TIM_SET_COUNTER(&htim4, 0);

	//	  elapsedTimeNoPrint = elapsedTime - sample_start_time;

//		  printf("%d\n\r", (int) elapsedTime);
		  printf("\n");
		  //		  printf("%d\n\r", (int) elapsedTime);

	//	  printf("\n\r");



		}



	/* USER CODE END WHILE */

	/* USER CODE BEGIN 3 */

	}
}

