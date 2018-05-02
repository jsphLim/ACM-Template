


##### KMP算法
```cpp
/*
字符串是从0开始的
Next数组是从1开始的
*/
const int N = 1000002;
int next[N];
char S[N], T[N];
int slen, tlen;

void getNext()
{
    int j, k;
    j = 0; k = -1; next[0] = -1;
    while(j < tlen)
        if(k == -1 || T[j] == T[k])
            next[++j] = ++k;
        else
            k = next[k];

}
int KMP_Index()
{
    int i = 0, j = 0;
    getNext();

    while(i < slen && j < tlen)
    {
        if(j == -1 || S[i] == T[j])
        {
            i++; j++;
        }
        else
            j = next[j];
    }
    if(j == tlen)
        return i - tlen;
    else
        return -1;
}
/*
返回模式串在主串S中出现的次数
*/
int KMP_Count()
{
    int ans = 0;
    int i, j = 0;

    if(slen == 1 && tlen == 1)
    {
        if(S[0] == T[0])
            return 1;
        else
            return 0;
    }
    getNext();
    for(i = 0; i < slen; i++)
    {
        while(j > 0 && S[i] != T[j])
            j = next[j];
        if(S[i] == T[j])
            j++;
        if(j == tlen)
        {
            ans++;
            j = next[j];
        }
    }
    return ans;
}
int main()
{  
    int TT;
    int i, cc;
    cin>>TT;
    while(TT--)
    {
        cin>>S>>T;
        slen = strlen(S);
        tlen = strlen(T);
        cout<<"模式串T在主串S中首次出现的位置是: "<<KMP_Index()<<endl;
        cout<<"模式串T在主串S中出现的次数为: "<<KMP_Count()<<endl;
    }
    return 0;
}
```
附上strstr函数
```cpp
/*
 *  strstr函数
 *  功能：在串中查找指定字符串的第一次出现
 *  用法：char *strstr(char *strOne, char *strTwo);
 */
int main(int argc, const char * argv[])
{
    char strOne[] = "Borland International";
    char strTwo[] = "nation";
    char *ptr;
    ptr = strstr(strOne, strTwo);
    std::cout << ptr << '\n';
    return 0;
}
```
