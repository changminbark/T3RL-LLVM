int isqrt_approx(int n){ int x=n, y=(x+1)/2; if(n<=1) return n; while(y<x){ x=y; y=(x+n/x)/2; } return x; }
