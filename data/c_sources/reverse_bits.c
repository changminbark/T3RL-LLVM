unsigned reverse_bits(unsigned x){ unsigned r=0; for(int i=0;i<32;i++){ r=(r<<1)|(x&1); x>>=1; } return r; }
