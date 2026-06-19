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

    // A4: scene_detect tests
    {   // identical frames -> no scene change
        const int w=4,h=2; uint8_t a[8]={10,20,30,40,50,60,70,80}, b[8]; for(int i=0;i<8;i++) b[i]=a[i];
        CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 64, 64, 1.0/255.0, 0.150) == false);
    }
    {   // hard cut: max difference everywhere -> scene change
        const int w=4,h=2; uint8_t a[8]; uint8_t b[8]; for(int i=0;i<8;i++){a[i]=0;b[i]=255;}
        // sum = 8*255*(1/255) = 8 ; /(64*64=4096) = 0.00195 < 0.15 -> NOT a scene at 64x64 padded
        CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 64, 64, 1.0/255.0, 0.150) == false);
        // but with padded area == unpadded (pw=4,ph=2 -> /8) -> 8/8=1.0 > 0.15 -> scene
        CHECK(rife_cpu::scene_detect(a, w, b, w, w, h, 4, 2, 1.0/255.0, 0.150) == true);
    }

    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
