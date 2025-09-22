#include <LiquidCrystal.h>

#ifndef _OLED12864_H_
#define _OLED12864_H_

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
 
#define OLED_RESET 4
 
//中文：转
static const unsigned char PROGMEM str_zhuan[] =
{ 
0x20,0x20,0x20,0x20,0x20,0x20,0xFD,0xFC,0x40,0x20,0x50,0x40,0x93,0xFE,0xFC,0x40,
0x10,0x80,0x11,0xFC,0x1C,0x04,0xF0,0x88,0x50,0x50,0x10,0x20,0x10,0x10,0x10,0x10
  };
 
//中文：向
static const unsigned char PROGMEM str_xiang[] =
{ 
0x02,0x00,0x04,0x00,0x08,0x00,0x7F,0xFC,0x40,0x04,0x40,0x04,0x47,0xC4,0x44,0x44,
0x44,0x44,0x44,0x44,0x44,0x44,0x47,0xC4,0x44,0x44,0x40,0x04,0x40,0x14,0x40,0x08
  };
 
//中文：速
static const unsigned char PROGMEM str_su[] =
{ 
0x00,0x40,0x20,0x40,0x17,0xFC,0x10,0x40,0x03,0xF8,0x02,0x48,0xF2,0x48,0x13,0xF8,
0x10,0xE0,0x11,0x50,0x12,0x48,0x14,0x44,0x10,0x40,0x28,0x00,0x47,0xFE,0x00,0x00
  };
 
// #if (SSD1306_LCDHEIGHT != 64)
// #error("Height incorrect, please fix Adafruit_SSD1306.h!");
// #endif
 

// class OLED12864 : public Adafruit_SSD1306 {
// 	public:
// 		OLED12864():Adafruit_SSD1306(OLED_RESET){}
		
// 		void init(){  
// 			begin(SSD1306_SWITCHCAPVCC, 0x3C); 	   
// 			setTextSize(2);             //设置字体大小
// 			setTextColor(WHITE);        //设置字体颜色白色
// 		}
		
// 		void clear(){ clearDisplay(); }
// 		void show(int y, int x, const String &s);
// 		void show(int y, int x, int num);
// 		void show(int y, int x, double num);
// 		void showCH(int yt, int xt, uint8_t bitmap[]);
// };

class OLED12864 : public Adafruit_SSD1306 {
  public:
    OLED12864():Adafruit_SSD1306(128, 64, &Wire, OLED_RESET) {}
    
    void init() {  
      begin(SSD1306_SWITCHCAPVCC, 0x3C);       
      setTextSize(2);             // Set text size
      setTextColor(SSD1306_WHITE); // Set text color to white
    }
    
    void clear() { clearDisplay(); }
    void show(int y, int x, const String &s);
    void show(int y, int x, int num);
    void show(int y, int x, double num);
    void showCH(int yt, int xt, uint8_t bitmap[]);
};



#endif