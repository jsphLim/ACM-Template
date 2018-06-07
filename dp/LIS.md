##### 最长有序子序列O(nlogn)
```cpp
/*
 *  递增（默认）
 *  递减
 *  非递增
 *  非递减 (1)>= && <  (2)<  (3)>=
 */
const int MAXN = 1001;

int a[MAXN], f[MAXN], d[MAXN];   //  d[i] 用于记录 a[0...i] 以 a[i] 结尾的最大长度

int bsearch(const int *f, int size, const int &a)
{
    int l = 0, r = size - 1;
    while (l <= r)
    {
        int mid = (l + r) / 2;
        if (a > f[mid - 1] && a <= f[mid])  //  (1)
        {
            return mid;
        }
        else if (a < f[mid])
        {
            r = mid - 1;
        }
        else
        {
            l = mid + 1;
        }
    }
    return -1;
}

int LIS(const int *a, const int &n)
{
    int i, j, size = 1;
    f[0] = a[0];
    d[0] = 1;
    for (i = 1; i < n; ++i)
    {
        if (a[i] <= f[0])               //  (2)
        {
            j = 0;
        }
        else if (a[i] > f[size - 1])    //  (3)
        {
            j = size++;
        }
        else
        {
            j = bsearch(f, size, a[i]);
        }
        f[j] = a[i];
        d[i] = j + 1;
    }
    return size;
}

int main()
{
    int i, n;
    while (scanf("%d", &n) != EOF)
    {
        for (i = 0; i < n; ++i)
        {
            scanf("%d", &a[i]);
        }
        printf("%d\n", LIS(a, n));      // 求最大递增 / 上升子序列(如果为最大非降子序列,只需把上面的注释部分给与替换)
    }
    return 0;
}
```
