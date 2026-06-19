// src/rife_selftest.cpp — standalone CPU unit tests (no GPU). Build: g++ -std=c++17.
#include "rife_cpu.h"
#include <cstdio>
#include <cstdlib>
static int g_fail = 0;
#define CHECK(cond) do { if(!(cond)){ printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while(0)
int main() {
    CHECK(1 + 1 == 2);
    CHECK(rife_cpu::model_name(414, false)  == "rife_v4.14");
    CHECK(rife_cpu::model_name(47,  false)  == "rife_v4.7");
    CHECK(rife_cpu::model_name(4251, false) == "rife_v4.25_lite");      // 4-digit ending in 1
    CHECK(rife_cpu::model_name(414, true)   == "rife_v4.14_ensemble");
    CHECK(rife_cpu::model_name(4251, true)  == "rife_v4.25_lite_ensemble");
    CHECK(rife_cpu::model_name(4,   false)  == "");                     // <2 digits → invalid
    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
