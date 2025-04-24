#include <stdio.h>

void pred_period(const int len, int*x, int*y, const float per)
{
    unsigned long upper,lower;
    for (int i=0;i<len;i++){
        upper = (unsigned long)(x[i] * (1+per));
        lower = (unsigned long)(x[i] * (1-per));

        for (int j=i;j<len;j++){
            if (upper < x[j]){
                y[i] = j-i;
                break;
            }
            else if (lower > x[j]){
                y[i] = i-j;
                break;
            }
        }
    }
}

void price_barrier(const int len, int* x, int* y, const int period, const float per )
{
    unsigned long upper,lower;
    for (int i=0;i<len;i++){
        upper = (unsigned long)(x[i] * (1+per));
        lower = (unsigned long)(x[i] * (1-per));

        for (int j=i;j<len;j++){
            if (upper < x[j]){
                y[i] = 1;
                break;
            }
            else if (lower > x[j]){
                y[i] = -1;
                break;
            }
        }
    }
}