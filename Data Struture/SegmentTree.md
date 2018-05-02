##### Segment Tree

```cpp
#define lson l , m , rt << 1  
#define rson m + 1 , r , rt << 1 | 1  
const int maxn = 55555;  
int sum[maxn<<2];  
void PushUP(int rt) {  
       sum[rt] = sum[rt<<1] + sum[rt<<1|1];   //求和操作 可更改(极值)
}  
void build(int l,int r,int rt) {  
       if (l == r) {  
              scanf("%d",&sum[rt]);  
              return ;  
       }  
       int m = (l + r) >> 1;  
       build(lson);  
       build(rson);  
       PushUP(rt);  
}  
void update(int p,int add,int l,int r,int rt) {  
       if (l == r) {  
              sum[rt] += add;  
              return ;  
       }  
       int m = (l + r) >> 1;  
       if (p <= m) update(p , add , lson);  
       else update(p , add , rson);  
       PushUP(rt);  
}  
int query(int L,int R,int l,int r,int rt) {  
       if (L <= l && r <= R) {  
              return sum[rt];  
       }  
       int m = (l + r) >> 1;  
       int ret = 0;  
       if (L <= m) ret += query(L , R , lson);  
       if (R > m) ret += query(L , R , rson);  
       return ret;  
}  
int main() {  
       int T , n;  
       scanf("%d",&T);  
       for (int cas = 1 ; cas <= T ; cas ++) {  
              printf("Case %d:\n",cas);  
              scanf("%d",&n);  
              build(1 , n , 1);  
              char op[10];  
              while (scanf("%s",op)) {  
                     if (op[0] == 'E') break;  
                     int a , b;  
                     scanf("%d%d",&a,&b);  
                     if (op[0] == 'Q') printf("%d\n",query(a , b , 1 , n , 1));  
                     else if (op[0] == 'S') update(a , -b , 1 , n , 1);  
                     else update(a , b , 1 , n , 1);  
              }  
       }  
       return 0;  
}  
```
#### 线段树扫描线
```cpp
const int maxn=1e5+50;
struct Type
{
    double l,r,h;  //以横坐标建立线段树，纵坐标为高度
    int cur;   //cur=-1说明是出边，cur=1说明是入边

    void sets( double x1, double x2, double h, int cur )
    {
        this->l=x1; this->r=x2; this->h=h; this->cur=cur;
    }
    bool operator < ( const Type& a )const
    {
        return h<a.h;
    }
};
Type seg[maxn];
double sumv[maxn*4]; //用来维护线段并
int cntv[maxn*4]; //用来维护当前节点代表的区间被覆盖了几次
double point[maxn*2];//用来维护离散化后的端点
int n;
int uniq(int k)
{
    int m=1;
    sort( point+1,point+1+k );
    for( int i=1;i<k;i++ )
    {
        if( point[i]!=point[i+1] )point[++m]=point[i+1];
    }
    return m;
}

void build( int O,int L,int R )
{
    if(L==R){ sumv[O]=0; cntv[O]=0; }
    else
    {
        int mid=(L+R)/2;
        build( O*2,L,mid );
        build( O*2+1,mid+1,R );
        sumv[O]=0; cntv[O]=0;
    }
}


void maintain( int O, int L, int R )
{
    if( cntv[O] )
    {
        sumv[O]=point[R+1]-point[L];
    }
    else if( L<R )
    {
        sumv[O]=sumv[O*2]+sumv[O*2+1];
       // cntv[O]=min( cntv[O*2],cntv[O*2+1] );
    }
    else { sumv[O]=0; cntv[O]=0; }
}

void pushdown( int O )
{
    if( cntv[O] )
    {
        cntv[O*2]=cntv[O*2+1]=cntv[O];
        cntv[O]=0;
        sumv[O]=0;
    }
}

void update( int O, int L, int R, int qL, int qR,int op )
{
    if( qL<=L && R<=qR )
    {
        cntv[O]+=op;
    }
    else
    {
       // pushdown(O);    //pushdown其实是不需要的，因为我们遇到一个cnt就可以返回整段信息，也就是说，更深的cnt信息不需要考虑，所以cnt没必要下传
        int mid=(L+R)/2;
        if( qL<=mid )update( O*2,L,mid,qL,qR,op );
        if( qR>mid )update( O*2+1,mid+1,R,qL,qR,op );
    }
    maintain( O,L,R );//重新计算sumv[O];
}

int main()
{
    int kase=0;
    while( ~scanf("%d",&n) &&n )
    {
        int k=0;
        for( int i=1; i<=n; i++ )
        {
            double x1,x2,y1,y2;
            scanf( "%lf%lf%lf%lf",&x1,&y1,&x2,&y2 );
            seg[++k].sets( x1,x2,y1,-1 );
            point[k]=x1;
            seg[++k].sets( x1,x2,y2,1 );
            point[k]=x2;
        }
        int m=uniq(k);  //对端点进行了离散化
        sort( seg+1,seg+1+k ); //所有k条线段从低到高排序
        build( 1,1,m );
        double ans=0;
        for( int i=1; i<k; i++ )
        {
            int L=lower_bound( point+1,point+1+m,seg[i].l )-point;
            int R=lower_bound( point+1,point+1+m,seg[i].r )-point-1;
            update( 1,1,m,L,R,seg[i].cur );
            ans+=sumv[1]*( seg[i+1].h-seg[i].h );
        }
        printf("Test case #%d\nTotal explored area: %.2lf\n\n",++kase,ans);
    }

    return 0;
}
```
