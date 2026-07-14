int color_blend(int a,int b,int alpha){ int inv=255-alpha; return (a*alpha + b*inv)/255; }
