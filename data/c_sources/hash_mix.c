unsigned hash_mix(unsigned x){ x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; return x; }
