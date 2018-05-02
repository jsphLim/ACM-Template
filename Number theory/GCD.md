#### GCD

```cpp
int gcd(int x, int y) {
  if (!x || !y) {
    return x > y ? x : y;
  }
  for (int t; t = x % y, t; x = y, y = t) ;
  return y;
}
```

#### Extern_GCD
```cpp
/* * 求x，y使得gcd(a, b) = a * x + b * y; */
int extgcd(int a, int b, int &x, int &y) {
  if (b == 0) { x = 1; y = 0; return a; }
  int d = extgcd(b, a % b, x, y);
  int t = x; x = y; y = t - a / b * y;
  return d;
}
```
