unsigned pack_rgba(unsigned r,unsigned g,unsigned b,unsigned a){ return (r&0xFF)|((g&0xFF)<<8)|((b&0xFF)<<16)|((a&0xFF)<<24); }
