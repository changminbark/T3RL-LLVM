unsigned swap_bits(unsigned x){ return ((x & 0x00FF) << 8) | ((x & 0xFF00) >> 8); }
