#include "printing.h"
//#include <stdio.h>

PUTCHAR_PROTOTYPE
{
  HAL_UART_Transmit(&huart3, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
//  HAL_UART_Transmit_IT(&huart3, (uint8_t *)&ch, 1);

  return ch;
}
