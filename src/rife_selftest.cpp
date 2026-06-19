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

    // A3: geometry tests
    { auto g = rife_cpu::geometry(1920, 1080);
      CHECK(g.pw == 1920);                       // 1920 already mod-64
      CHECK(g.ph == 1088);                       // 1080 -> 1088
      CHECK(g.pad_l == 0);
      CHECK(g.pad_t == (((1088-1080)/2) & ~1));  // = 4
    }
    { auto g = rife_cpu::geometry(854, 480);
      CHECK(g.pw == 896);                        // (854+63)/64*64
      CHECK(g.ph == 512);                        // (480+63)/64*64
      CHECK(g.pad_l == (((896-854)/2) & ~1));    // 21 -> 20
      CHECK(g.pad_t == (((512-480)/2) & ~1));    // 16
    }

    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
