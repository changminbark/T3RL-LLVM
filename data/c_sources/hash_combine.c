unsigned hash_combine(unsigned a,unsigned b){ unsigned h=a; h^=b + 0x9e3779b9u + (h<<6) + (h>>2); return h; }
