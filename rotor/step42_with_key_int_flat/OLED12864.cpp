#include "OLED12864.h"

void OLED12864:: show(int y, int x, const String &s){
	   setCursor(12*x,16*y);             //设置字体的起始位置
	   println(s);   //输出字符并换行
	   //display();                  //显示以上
}

void OLED12864::  show(int y, int x, int num){
	   setCursor(12*x,16*y);             //设置字体的起始位置
	   println(num);   //输出字符并换行
	   //display();                  //显示以上
	
}

void OLED12864::  show(int y, int x, double num){
	   setCursor(12*x,16*y);             //设置字体的起始位置
	   println(num);   //输出字符并换行
	   //display();                  //显示以上
}

void OLED12864:: showCH(int yt, int xt, uint8_t bitmap[]){
	
   int16_t w=16;
   int16_t h=16; 
   uint16_t color=1;
   int16_t x=16*xt;
   int16_t y=16*yt; 
   
    int16_t byteWidth = (w + 7) / 8; // Bitmap scanline pad = whole byte
    uint8_t byte = 0;

    startWrite();
    for(int16_t j=0; j<h; j++, y++) {
        for(int16_t i=0; i<w; i++) {
            if(i & 7) byte <<= 1;
            else      byte   = pgm_read_byte(&bitmap[j * byteWidth + i / 8]);
            if(byte & 0x80) writePixel(x+i, y, color);
        }
    }
    endWrite();
	
	//display(); 
	
	//Serial.println("zwc");
}

