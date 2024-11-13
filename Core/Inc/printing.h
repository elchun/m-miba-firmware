#include <stdio.h>
#include "stm32h7xx_hal.h"
#include "usart.h"

// printf code from "https://forum.digikey.com/t/easily-use-printf-on-stm32/20157"
#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif
