##### 1.最优三角剖分
```cpp
#define N 6        //顶点数/边数
int weight[][N] = {{0,2,2,3,1,4},{2,0,1,5,2,3},{2,1,0,2,1,4},{3,5,2,0,6,2},
{1,2,1,6,0,1},{4,3,4,2,1,0}};
int t[N][N];    //t[i][j]表示多边形{Vi-1VkVj}的最优权值
int s[N][N];    //s[i][j]记录Vi-1到Vj最优三角剖分的中间点K
int get_weight(const int a, const int b, const int c)
{
    return weight[a][b] + weight[b][c] + weight[c][a];
}
void minest_weight_val()
{
    int i,r,k,j;
    int min;

    for (i = 1; i < N; i++)
    {
        t[i][i] = 0;
    }

    for (r = 2; r < N; r++)
    {
        for (i = 1; i < N-r+1; i++)
        {
            j = i + r -1;
            min = 9999;        //假设最小值
            for (k = i; k < j; k++)
            {
                t[i][j] = t[i][k] + t[k+1][j] + get_weight(i-1,k,j);
                if (t[i][j] < min)        //判断是否是最小值
                {
                    min = t[i][j];
                    s[i][j] = k;    
                }
            }
            t[i][j] = min;        //取得多边形{Vi-1，Vj}的划分三角最小权值
        }
    }
}
void back_track(int a, int b)
{
    if (a == b) return;
    back_track(a,s[a][b]);
    back_track(s[a][b]+1,b);    //记得这是要加一
    printf("最优三角: V%d V%d V%d.\n",a-1,s[a][b],b);
}
int main()
{
    minest_weight_val();
    printf("result:%d\n",t[1][N-1]);
    back_track(1,5);
    return 0;
}
```
#### 四边形优化
```cpp
#include <cstdio>  
#include <queue>  
#include <cstring>  
#include <algorithm>  
using namespace std;  
#define mst(a,b) memset((a),(b),sizeof(a))  
#define rush() int T;scanf("%d",&T);while(T--)  

typedef long long ll;  
const int maxn = 205;  
const ll mod = 1e9+7;  
const ll INF = 1e18;  
const double eps = 1e-9;  

int n,x;  
int sum[maxn];  
int dp[maxn][maxn];  
int s[maxn][maxn];  

int main()  
{  
    while(~scanf("%d",&n))  
    {  
        sum[0]=0;  
        mst(dp,0x3f);  
        for(int i=1;i<=n;i++)  
        {  
            scanf("%d",&x);  
            sum[i]=sum[i-1]+x;  
            dp[i][i]=0;  
            s[i][i]=i;  
        }  
        for(int len=2;len<=n;len++)  
        for(int i=1;i<=n;i++)  
        {  
            int j=i+len-1;  
            if(j>n) continue;  
            for(int k=s[i][j-1];k<=s[i+1][j];k++)  
            {  
                if(dp[i][k]+dp[k+1][j]+sum[j]-sum[i-1]<dp[i][j])  
                {  
                    dp[i][j]=dp[i][k]+dp[k+1][j]+sum[j]-sum[i-1];  
                    s[i][j]=k;  
                }  
            }  
        }  
        printf("%d\n",dp[1][n]);  
    }  
    return 0;  
}  
```
