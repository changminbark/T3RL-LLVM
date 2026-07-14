unsigned bcd_encode(unsigned n){ return ((n/1000%10)<<12)|((n/100%10)<<8)|((n/10%10)<<4)|(n%10); }
