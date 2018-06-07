```cpp
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
using namespace std;
typedef long long ll;
const int N=5e4+5;
inline int read(){
    char c=getchar();int x=0,f=1;
    while(c<'0'||c>'9'){if(c=='-')f=-1;c=getchar();}
    while(c>='0'&&c<='9'){x=x*10+c-'0';c=getchar();}
    return x*f;
}
int n;
struct Operation{
    int a,b,c,d;
    bool flag;
}a[N],t1[N],t2[N];
int c[N];
inline int lowbit(int x){return x&-x;}
inline void add(int p,int v){for(;p<=n;p+=lowbit(p)) c[p]+=v;}
inline int sum(int p){
    int re=0;
    for(;p;p-=lowbit(p)) re+=c[p];
    return re;
}
int ans;
void CDQ2(int l,int r){
    if(l==r) return;
    int mid=(l+r)>>1;
    CDQ2(l,mid);CDQ2(mid+1,r);
    int i=l,j=mid+1,p=l;
    Operation *a=t1,*t=t2;
    while(i<=mid||j<=r){
        if(j>r||(i<=mid&&a[i].c<a[j].c)){
            if(a[i].flag) add(a[i].d,1);
            t[p++]=a[i++];
        }else{
            if(!a[j].flag) ans+=sum(a[j].d);
            t[p++]=a[j++];
        }
    }
    for(int i=l;i<=mid;i++) if(a[i].flag) add(a[i].d,-1);
    for(int i=l;i<=r;i++) a[i]=t[i];
}
void CDQ(int l,int r){
    if(l==r) return;
    int mid=(l+r)>>1;
    CDQ(l,mid);CDQ(mid+1,r);
    int i=l,j=mid+1,p=l;
    Operation *t=t1;
    while(i<=mid||j<=r){
        if(j>r||(i<=mid&&a[i].b<a[j].b)) (t[p++]=a[i++]).flag=1;
        else (t[p++]=a[j++]).flag=0;
    }
    for(int i=l;i<=r;i++) a[i]=t[i];
    CDQ2(l,r);
}
int main(){
    freopen("partial_order.in","r",stdin);
    freopen("partial_order.out","w",stdout);
    n=read();
    for(int i=1;i<=n;i++) a[i].b=read();
    for(int i=1;i<=n;i++) a[i].c=read();
    for(int i=1;i<=n;i++) a[i].d=read(),a[i].a=i;
    CDQ(1,n);
    printf("%d",ans);
}
```
