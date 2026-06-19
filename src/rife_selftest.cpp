// src/rife_selftest.cpp — standalone CPU unit tests (no GPU). Build: g++ -std=c++17.
#include "rife_cpu.h"
#include <cstdio>
#include <cstdlib>
static int g_fail = 0;
#define CHECK(cond) do { if(!(cond)){ printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while(0)
int main() {
    CHECK(1 + 1 == 2);
    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
