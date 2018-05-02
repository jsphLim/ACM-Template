
##### 欧拉函数
```cpp
/* * 单独求解的本质是公式的应用 */
unsigned euler(unsigned x) {
  unsigned i, res = x;
  // unsigned == unsigned int
  for (i = 2; i < (int)sqrt(x * 1.0) + 1; i++) {
     if (!(x % i)) {
        res = res / i * (i - 1);
         while (!(x % i)) {
            x /= i;
            // 保证i一定是素数
        }
      }
    }
      if (x > 1) {
        res = res / x * (x - 1);
      }
       return res;
}
```
#### 线性筛
```cpp
/* * 同时得到欧拉函数和素数表 */
const int MAXN = 10000000;
bool check[MAXN + 10];
int phi[MAXN + 10];
int prime[MAXN + 10];
int tot; // 素数个数
void phi_and_prime_table(int N) {
  memset(check, false, sizeof(check));
  phi[1] = 1; tot = 0;
  for (int i = 2; i <= N; i++) {
     if (!check[i]) {
       prime[tot++] = i;
       phi[i] = i - 1;
      }
      for (int j = 0; j < tot; j++) {
        if (i * prime[j] > N) { break; }
        check[i * prime[j]] = true;
        if (i % prime[j] == 0) {
           phi[i * prime[j]] = phi[i] * prime[j]; break;
        } else {
             phi[i * prime[j]] = phi[i] * (prime[j] - 1);
        }
      }
  }
  return ;
}

```
