#include <stdio.h>

void pred_period(const int len, int*x, int*y, const float per)
{
    unsigned long upper,lower;
    printf("len: %d\n",len);
    printf("x first/last: %d / %d\n",x[0],x[len-1]);
    printf("y first/last: %d / %d\n",y[0],y[len-1]);
    printf("per: %f\n",per);
    for (int i=0;i<len;i++){
        upper = (unsigned long)(x[i] * (1+per));
        lower = (unsigned long)(x[i] * (1-per));

        for (int j=i+1;j<len;j++){
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
    printf("   len: %d\n",len);
    printf("   x first/last: %d / %d\n",x[0],x[len-1]);
    printf("   y first/last: %d / %d\n",y[0],y[len-1]);
    printf("   period: %d\n",period);
    printf("   per: %f\n",per);

    for (int i=0;i<len;i++){
        upper = (unsigned long)(x[i] * (1+per));
        lower = (unsigned long)(x[i] * (1-per));

        int max_j = (i + period) < len ? (i + period) : len;

        for (int j=i+1;j<max_j;j++){
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
    printf("    end c code\n");
}


void price_barrier_volat(const int len, int* x, int* y, float*per, const int period)
{
    unsigned long upper,lower;
    printf("   len: %d\n",len);
    printf("   x first/last: %d / %d\n",x[0],x[len-1]);
    printf("   y first/last: %d / %d\n",y[0],y[len-1]);
    printf("   per first/last: %f / %f\n",per[0],per[len-1]);
    printf("   period: %d\n",period);

    for (int i=0;i<len;i++){
        upper = (unsigned long)(x[i] * (1+per[i]));
        lower = (unsigned long)(x[i] * (1-per[i]));

        int max_j = (i + period) < len ? (i + period) : len;

        for (int j=i+1;j<max_j;j++){
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
    printf("    end c code\n");
}