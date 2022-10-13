/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2022 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "string.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "ai_datatypes_defines.h"
#include "ai_platform.h"

#include "lstm.h"
#include "lstm_data.h"
#include "stm32f767xx.h"
#include "stdlib.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
struct Out_param {
	float mean;
	float var;
};
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define _getCounter(TIMx) LL_TIM_GetCounter(TIMx)
#define _startCounter(TIMx) configCounter(TIMx)
#define _stopCounter(TIMx) LL_TIM_DisableCounter(TIMx)
#define _resetCounter(TIMx,overflow) resetCounter(TIMx, overflow)
#define SIZEOF(a) (sizeof(a)/ sizeof(a[0]))
#define Max(a,b) (a>b?a:b)
//#define RAND_MAX 32767
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
#if defined ( __ICCARM__ ) /*!< IAR Compiler */
#pragma location=0x2007c000
ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
#pragma location=0x2007c0a0
ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __CC_ARM )  /* MDK ARM Compiler */

__attribute__((at(0x2007c000))) ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
__attribute__((at(0x2007c0a0))) ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __GNUC__ ) /* GNU Compiler */
ETH_DMADescTypeDef DMARxDscrTab[ETH_RX_DESC_CNT] __attribute__((section(".RxDecripSection"))); /* Ethernet Rx DMA Descriptors */
ETH_DMADescTypeDef DMATxDscrTab[ETH_TX_DESC_CNT] __attribute__((section(".TxDecripSection"))); /* Ethernet Tx DMA Descriptors */

#endif

ETH_TxPacketConfig TxConfig;

CRC_HandleTypeDef hcrc;

ETH_HandleTypeDef heth;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */
int f_apb1clk = 108e6;
int overflow_1 = 0;
int overflow_2 = 0;
float timeX = 0.f;
float timeY = 0.f;

char buf[50];
int buf_len = 0;
ai_error ai_err;
ai_i32 nbatch;
uint32_t timestamp;
float y_val;

//variables for cusum
struct Out_param outparam;
float cp, cm = 0.f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ETH_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_CRC_Init(void);
static void MX_TIM14_Init(void);
static void MX_TIM13_Init(void);
static void MX_RNG_Init(void);
/* USER CODE BEGIN PFP */
uint8_t sendDebugMsg(char *msg);
void configCounter(TIM_TypeDef *TIMx);
float getTimeus(TIM_TypeDef *TIMx, int overflow);
void resetCounter(TIM_TypeDef *TIMx, int *overflow);
float PCATransformTest(int in_dim, const int out_dim, int cycles);
void randomInput(float *inBuffer, int count);
void randomMatrix(float *PCABuffer, int row, int column);
void calcPCA(float *output, float *input, float *matrix, int row, int column);
float modelRun(float *input, int length, float *prediction);
void normalizationPred(float *pred, struct Out_param outparam);
float calcCP(float *pred, struct Out_param outparam, float w);
float calcCM(float *pred, struct Out_param outparam, float w);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {
	/* USER CODE BEGIN 1 */

//	ai_float mu[3] = { 1.39730766, 41.25205743, 16.35722421 };
//	ai_float std[3] = { 0.06754268, 0.07502506, 0.03128953 };
// if we don't use PCA
	/* USER CODE END 1 */

	/* Enable I-Cache---------------------------------------------------------*/
	SCB_EnableICache();

	/* Enable D-Cache---------------------------------------------------------*/
	SCB_EnableDCache();

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_ETH_Init();
	MX_USART3_UART_Init();
	MX_USB_OTG_FS_PCD_Init();
	MX_CRC_Init();
	MX_TIM14_Init();
	MX_TIM13_Init();
	MX_RNG_Init();
	/* USER CODE BEGIN 2 */
	outparam.mean = 0.0338695f;
	outparam.var = 0.00167553f;
	float w = outparam.var;
	static int in_shape = 3;
	static int timestep = 4;
	static int pca_dimension = 4;
//	float t1 = PCATransformTest(in_shape, pca_dimension, 100); // This function can be used with only 3 param, no need of input array.

	float in[in_shape][timestep];
	randomInput(in, in_shape * timestep);

//	float pca_matrix[pca_dimension][in_shape];
//	randomMatrix(pca_matrix, pca_dimension, in_shape);

//	float out[pca_dimension];
	float pred = 0.f;
//	calcPCA(out, in, pca_matrix, pca_dimension, in_shape);
//	for (int i = 0; i < 100; i++) {
//		timeX += modelRun(in, SIZEOF(in), &pred);
//	}
//	timeX = timeX / 100;
	timeX = modelRun(in, SIZEOF(in), &pred);
	normalizationPred(&pred, outparam);
	_startCounter(TIM13);
	float cp = calcCP(&pred, outparam, w);
	float cm = calcCM(&pred, outparam, w);
	_stopCounter(TIM13);
	timeX = getTimeus(TIM13, overflow_2);
	resetCounter(TIM13, &overflow_2);
	if (cp > 5 || cm > 5) {

	}
//	float pca_matrix[2][6] = { { -0.44280883, -0.36813074, -0.48279947,
//			-0.48280469, -0.44967144, -0.44967144 }, { -0.47070612, 0.74485537,
//			0.13377964, 0.13367727, -0.43342776, -0.44967144 } };

	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
	RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };

	/** Configure LSE Drive Capability
	 */
	HAL_PWR_EnableBkUpAccess();

	/** Configure the main internal regulator output voltage
	 */
	__HAL_RCC_PWR_CLK_ENABLE();
	__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLM = 4;
	RCC_OscInitStruct.PLL.PLLN = 216;
	RCC_OscInitStruct.PLL.PLLP = 2;
	RCC_OscInitStruct.PLL.PLLQ = 9;
	RCC_OscInitStruct.PLL.PLLR = 2;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/** Activate the Over-Drive mode
	 */
	if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK) {
		Error_Handler();
	}
}

/**
 * @brief CRC Initialization Function
 * @param None
 * @retval None
 */
static void MX_CRC_Init(void) {

	/* USER CODE BEGIN CRC_Init 0 */

	/* USER CODE END CRC_Init 0 */

	/* USER CODE BEGIN CRC_Init 1 */

	/* USER CODE END CRC_Init 1 */
	hcrc.Instance = CRC;
	hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
	hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
	hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
	hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
	hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
	if (HAL_CRC_Init(&hcrc) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN CRC_Init 2 */

	/* USER CODE END CRC_Init 2 */

}

/**
 * @brief ETH Initialization Function
 * @param None
 * @retval None
 */
static void MX_ETH_Init(void) {

	/* USER CODE BEGIN ETH_Init 0 */

	/* USER CODE END ETH_Init 0 */

	static uint8_t MACAddr[6];

	/* USER CODE BEGIN ETH_Init 1 */

	/* USER CODE END ETH_Init 1 */
	heth.Instance = ETH;
	MACAddr[0] = 0x00;
	MACAddr[1] = 0x80;
	MACAddr[2] = 0xE1;
	MACAddr[3] = 0x00;
	MACAddr[4] = 0x00;
	MACAddr[5] = 0x00;
	heth.Init.MACAddr = &MACAddr[0];
	heth.Init.MediaInterface = HAL_ETH_RMII_MODE;
	heth.Init.TxDesc = DMATxDscrTab;
	heth.Init.RxDesc = DMARxDscrTab;
	heth.Init.RxBuffLen = 1524;

	/* USER CODE BEGIN MACADDRESS */

	/* USER CODE END MACADDRESS */

	if (HAL_ETH_Init(&heth) != HAL_OK) {
		Error_Handler();
	}

	memset(&TxConfig, 0, sizeof(ETH_TxPacketConfig));
	TxConfig.Attributes = ETH_TX_PACKETS_FEATURES_CSUM
			| ETH_TX_PACKETS_FEATURES_CRCPAD;
	TxConfig.ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC;
	TxConfig.CRCPadCtrl = ETH_CRC_PAD_INSERT;
	/* USER CODE BEGIN ETH_Init 2 */

	/* USER CODE END ETH_Init 2 */

}

/**
 * @brief RNG Initialization Function
 * @param None
 * @retval None
 */
static void MX_RNG_Init(void) {

	/* USER CODE BEGIN RNG_Init 0 */

	/* USER CODE END RNG_Init 0 */

	RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };

	/** Initializes the peripherals clock
	 */
	PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_CLK48;
	PeriphClkInitStruct.Clk48ClockSelection = RCC_CLK48SOURCE_PLL;
	if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/* Peripheral clock enable */
	LL_AHB2_GRP1_EnableClock(LL_AHB2_GRP1_PERIPH_RNG);

	/* USER CODE BEGIN RNG_Init 1 */

	/* USER CODE END RNG_Init 1 */
	LL_RNG_Enable(RNG);
	/* USER CODE BEGIN RNG_Init 2 */

	/* USER CODE END RNG_Init 2 */

}

/**
 * @brief TIM13 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM13_Init(void) {

	/* USER CODE BEGIN TIM13_Init 0 */

	/* USER CODE END TIM13_Init 0 */

	LL_TIM_InitTypeDef TIM_InitStruct = { 0 };

	/* Peripheral clock enable */
	LL_APB1_GRP1_EnableClock(LL_APB1_GRP1_PERIPH_TIM13);

	/* TIM13 interrupt Init */
	NVIC_SetPriority(TIM8_UP_TIM13_IRQn,
			NVIC_EncodePriority(NVIC_GetPriorityGrouping(), 0, 0));
	NVIC_EnableIRQ(TIM8_UP_TIM13_IRQn);

	/* USER CODE BEGIN TIM13_Init 1 */

	/* USER CODE END TIM13_Init 1 */
	TIM_InitStruct.Prescaler = 0;
	TIM_InitStruct.CounterMode = LL_TIM_COUNTERMODE_UP;
	TIM_InitStruct.Autoreload = 65535;
	TIM_InitStruct.ClockDivision = LL_TIM_CLOCKDIVISION_DIV1;
	LL_TIM_Init(TIM13, &TIM_InitStruct);
	LL_TIM_DisableARRPreload(TIM13);
	/* USER CODE BEGIN TIM13_Init 2 */

	/* USER CODE END TIM13_Init 2 */

}

/**
 * @brief TIM14 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM14_Init(void) {

	/* USER CODE BEGIN TIM14_Init 0 */

	/* USER CODE END TIM14_Init 0 */

	LL_TIM_InitTypeDef TIM_InitStruct = { 0 };

	/* Peripheral clock enable */
	LL_APB1_GRP1_EnableClock(LL_APB1_GRP1_PERIPH_TIM14);

	/* TIM14 interrupt Init */
	NVIC_SetPriority(TIM8_TRG_COM_TIM14_IRQn,
			NVIC_EncodePriority(NVIC_GetPriorityGrouping(), 0, 0));
	NVIC_EnableIRQ(TIM8_TRG_COM_TIM14_IRQn);

	/* USER CODE BEGIN TIM14_Init 1 */

	/* USER CODE END TIM14_Init 1 */
	TIM_InitStruct.Prescaler = 0;
	TIM_InitStruct.CounterMode = LL_TIM_COUNTERMODE_UP;
	TIM_InitStruct.Autoreload = 65535;
	TIM_InitStruct.ClockDivision = LL_TIM_CLOCKDIVISION_DIV1;
	LL_TIM_Init(TIM14, &TIM_InitStruct);
	LL_TIM_DisableARRPreload(TIM14);
	/* USER CODE BEGIN TIM14_Init 2 */

	/* USER CODE END TIM14_Init 2 */

}

/**
 * @brief USART3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART3_UART_Init(void) {

	/* USER CODE BEGIN USART3_Init 0 */

	/* USER CODE END USART3_Init 0 */

	LL_USART_InitTypeDef USART_InitStruct = { 0 };

	LL_GPIO_InitTypeDef GPIO_InitStruct = { 0 };
	RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };

	/** Initializes the peripherals clock
	 */
	PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART3;
	PeriphClkInitStruct.Usart3ClockSelection = RCC_USART3CLKSOURCE_PCLK1;
	if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/* Peripheral clock enable */
	LL_APB1_GRP1_EnableClock(LL_APB1_GRP1_PERIPH_USART3);

	LL_AHB1_GRP1_EnableClock(LL_AHB1_GRP1_PERIPH_GPIOD);
	/**USART3 GPIO Configuration
	 PD8   ------> USART3_TX
	 PD9   ------> USART3_RX
	 */
	GPIO_InitStruct.Pin = LL_GPIO_PIN_8;
	GPIO_InitStruct.Mode = LL_GPIO_MODE_ALTERNATE;
	GPIO_InitStruct.Speed = LL_GPIO_SPEED_FREQ_VERY_HIGH;
	GPIO_InitStruct.OutputType = LL_GPIO_OUTPUT_PUSHPULL;
	GPIO_InitStruct.Pull = LL_GPIO_PULL_NO;
	GPIO_InitStruct.Alternate = LL_GPIO_AF_7;
	LL_GPIO_Init(GPIOD, &GPIO_InitStruct);

	GPIO_InitStruct.Pin = LL_GPIO_PIN_9;
	GPIO_InitStruct.Mode = LL_GPIO_MODE_ALTERNATE;
	GPIO_InitStruct.Speed = LL_GPIO_SPEED_FREQ_VERY_HIGH;
	GPIO_InitStruct.OutputType = LL_GPIO_OUTPUT_PUSHPULL;
	GPIO_InitStruct.Pull = LL_GPIO_PULL_NO;
	GPIO_InitStruct.Alternate = LL_GPIO_AF_7;
	LL_GPIO_Init(GPIOD, &GPIO_InitStruct);

	/* USER CODE BEGIN USART3_Init 1 */

	/* USER CODE END USART3_Init 1 */
	USART_InitStruct.BaudRate = 115200;
	USART_InitStruct.DataWidth = LL_USART_DATAWIDTH_8B;
	USART_InitStruct.StopBits = LL_USART_STOPBITS_1;
	USART_InitStruct.Parity = LL_USART_PARITY_NONE;
	USART_InitStruct.TransferDirection = LL_USART_DIRECTION_TX_RX;
	USART_InitStruct.HardwareFlowControl = LL_USART_HWCONTROL_NONE;
	USART_InitStruct.OverSampling = LL_USART_OVERSAMPLING_16;
	LL_USART_Init(USART3, &USART_InitStruct);
	LL_USART_ConfigAsyncMode(USART3);
	LL_USART_Enable(USART3);
	/* USER CODE BEGIN USART3_Init 2 */

	/* USER CODE END USART3_Init 2 */

}

/**
 * @brief USB_OTG_FS Initialization Function
 * @param None
 * @retval None
 */
static void MX_USB_OTG_FS_PCD_Init(void) {

	/* USER CODE BEGIN USB_OTG_FS_Init 0 */

	/* USER CODE END USB_OTG_FS_Init 0 */

	/* USER CODE BEGIN USB_OTG_FS_Init 1 */

	/* USER CODE END USB_OTG_FS_Init 1 */
	hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
	hpcd_USB_OTG_FS.Init.dev_endpoints = 6;
	hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
	hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
	hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
	hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
	hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
	if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN USB_OTG_FS_Init 2 */

	/* USER CODE END USB_OTG_FS_Init 2 */

}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
	GPIO_InitTypeDef GPIO_InitStruct = { 0 };

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOH_CLK_ENABLE();
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOG_CLK_ENABLE();

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(GPIOB, LD1_Pin | LD3_Pin | LD2_Pin, GPIO_PIN_RESET);

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin,
			GPIO_PIN_RESET);

	/*Configure GPIO pin : USER_Btn_Pin */
	GPIO_InitStruct.Pin = USER_Btn_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
	GPIO_InitStruct.Pin = LD1_Pin | LD3_Pin | LD2_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

	/*Configure GPIO pin : USB_PowerSwitchOn_Pin */
	GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pin : USB_OverCurrent_Pin */
	GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
uint8_t sendDebugMsg(char *msg) {
	uint8_t state = 0;
	LL_USART_ClearFlag_TC(USART3);
	uint8_t i = 0;
	do {
//		while(!LL_USART_IsActiveFlag_TXE(USART_PC));
		LL_USART_TransmitData8(USART3, *(msg + i));
		i++;
		while (!LL_USART_IsActiveFlag_TC(USART3))
			;
		LL_USART_ClearFlag_TC(USART3);
	} while (*(msg + i) != '\0');
	state = 1;
	memset(msg, 0, i);
	return state;
}

void configCounter(TIM_TypeDef *TIMx) {
	LL_TIM_EnableIT_UPDATE(TIMx);
	LL_TIM_ClearFlag_UPDATE(TIMx);
	LL_TIM_EnableCounter(TIMx);
}

float getTimeus(TIM_TypeDef *TIMx, int overflow) {
	uint32_t counter = _getCounter(TIMx);
	float dt = 1e6 / f_apb1clk;
	float overflowCounter = overflow * (LL_TIM_GetAutoReload(TIMx) + 1);
	return (counter + overflowCounter) * dt;
}

void resetCounter(TIM_TypeDef *TIMx, int *overflow) {
	LL_TIM_SetCounter(TIMx, 0x0);
	*overflow = 0;
}

float PCATransformTest(int in_dim, const int out_dim, int cycles) {
	float out[out_dim];
	float in[in_dim];
	float pca_matrix[out_dim][in_dim];
	float T_tot = 0.f;

	// init for in and pca_matrix
	for (int i = 0; i < in_dim; i++) {
		in[i] = (float) rand() / RAND_MAX;
	}
	for (int j = 0; j < out_dim; j++) {
		for (int i = 0; i < in_dim; i++) {
			pca_matrix[j][i] = (float) rand() / RAND_MAX;
		}
	}

	for (int c = 0; c < cycles; c++) {
		_startCounter(TIM13);
		for (int j = 0; j < out_dim; j++) {
			for (int i = 0; i < in_dim; i++) {
				out[j] += pca_matrix[j][i] * in[i];
			}
		}
		_stopCounter(TIM13);
		timeY = getTimeus(TIM13, overflow_2);
		_resetCounter(TIM13, &overflow_2);
		T_tot += timeY;
	}

	return T_tot / cycles;
}

void randomInput(float *inBuffer, int count) {
	for (int i = 0; i < count; i++) {
		inBuffer[i] = (float) rand() / RAND_MAX;
//		inBuffer[i] = 100*i; // For test usage
	}
}

void randomMatrix(float *PCABuffer, int row, int column) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			PCABuffer[i * column + j] = (float) rand() / RAND_MAX;
//			PCABuffer[i * column + j] = i * column + j; // For test usage
		}
	}
}

void calcPCA(float *output, float *input, float *matrix, int row, int column) {
	for (int i = 0; i < row; i++) {
		output[i] = 0;
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			output[i] += matrix[i * column + j] * input[j];
		}
	}
}

float modelRun(float *input, int length, float *prediction) {
	ai_u8 activations[AI_LSTM_DATA_ACTIVATIONS_SIZE];
	ai_i8 in_data[AI_LSTM_IN_1_SIZE_BYTES];
	ai_i8 out_data[AI_LSTM_OUT_1_SIZE_BYTES];

	ai_handle network = AI_HANDLE_NULL;
	ai_buffer *ai_input;
	ai_buffer *ai_output;
	ai_error err;
	ai_network_report report;
	float time = 0.f;

	const ai_handle acts[] = { activations };
	err = ai_lstm_create_and_init(&network, acts, NULL);
	if (err.type != AI_ERROR_NONE) {
		sprintf(buf, "\r\n\r\nai init_and_create error\n\r\n");
		sendDebugMsg(buf);
		return -1;
	} else {
		sprintf(buf, "\r\n\r\nai init_and_create finished\n\r\n");
		sendDebugMsg(buf);
	}
	if (ai_lstm_get_report(network, &report) != true) {
		sprintf(buf, "ai get report error\n");
		sendDebugMsg(buf);
		return -1;
	} else {
		sprintf(buf, "get report finished\r\n");
		sendDebugMsg(buf);
	}

	sprintf(buf, "Model name: %s\r\n", report.model_name);
	sendDebugMsg(buf);
	sprintf(buf, "Model signature: %s\r\n", report.model_signature);
	sendDebugMsg(buf);
	ai_input = &report.inputs[0];
	ai_output = &report.outputs[0];
	sprintf(buf, "input[0] : (%d, %d, %d)\r\n",
			AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_HEIGHT),
			AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_WIDTH),
			AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_CHANNEL));
	sendDebugMsg(buf);
	sprintf(buf, "output[0] : (%d, %d, %d)\r\n",
			AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_HEIGHT),
			AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_WIDTH),
			AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_CHANNEL));
	sendDebugMsg(buf);

	for (int i = 0; i < length; i++) {
		((ai_float*) in_data)[i] = (ai_float) input[i];
	}

//	((ai_float*) in_data)[1] = (ai_float) input[1];

	ai_i32 n_batch;
	ai_input = ai_lstm_inputs_get(network, NULL);
	ai_output = ai_lstm_outputs_get(network, NULL);
	/** @brief Set input/output buffer addresses */
	ai_input[0].data = AI_HANDLE_PTR(in_data);
	ai_output[0].data = AI_HANDLE_PTR(out_data);

	_startCounter(TIM14);
	n_batch = ai_lstm_run(network, &ai_input[0], &ai_output[0]);
	_stopCounter(TIM14);
	time = getTimeus(TIM14, overflow_1);
	resetCounter(TIM14, &overflow_1);
	if (n_batch != 1) {
		err = ai_lstm_get_error(network);
		printf("ai run error %d, %d\n", err.type, err.code);
		return -1;
	}
	prediction = ai_output[0].data;

	/** @brief Post-process the output results/predictions */
	printf("Inference output..\n");

//	sprintf(buf, "Avg Execution time: %f\r\n", T / 10);
//	sendDebugMsg(buf);
	return time;
}

void normalizationPred(float *pred, struct Out_param outparam) {
	*pred = (*pred - outparam.mean) / outparam.var;
}

float calcCP(float *pred, struct Out_param outparam, float w) {
	cp = Max(0, cp + *pred - w);
	return (cp);
}

float calcCM(float *pred, struct Out_param outparam, float w) {
	cm = Max(0, cm - *pred - w);
	return (cm);
}
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
