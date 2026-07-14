int count_bits_set(const unsigned *a, int n){ int c=0; for(int i=0;i<n;i++){ unsigned x=a[i]; while(x){ c+=x&1; x&=x-1; } } return c; }
