int matrix_trace(const int *m, int n){ int s=0; for(int i=0;i<n;i++) s+=m[i*n+i]; return s; }
