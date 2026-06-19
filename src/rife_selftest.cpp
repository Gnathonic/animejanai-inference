// src/rife_selftest.cpp — standalone CPU unit tests (no GPU). Build: g++ -std=c++17.
#include "rife_cpu.h"
#include "aji.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

    // A5: fill_consts — mesh/mul const channels 7-10 of the RIFE v1 input layout
    {   const int pw=4, ph=2; std::vector<float> t((size_t)11*pw*ph, -123.f);
        rife_cpu::fill_consts(t.data(), pw, ph);
        const int plane = pw*ph; auto at=[&](int ch,int x,int y){ return t[(size_t)ch*plane + y*pw + x]; };
        CHECK(at(7,0,0) == -1.0f);                          // meshX at x=0
        CHECK(at(7,pw-1,0) == 1.0f);                        // meshX at x=pw-1
        CHECK(at(8,0,ph-1) == 1.0f);                        // meshY at y=ph-1
        CHECK(std::abs(at(9,2,1) - (2.0f/(pw-1))) < 1e-6f); // mulW constant
        CHECK(std::abs(at(10,1,0) - (2.0f/(ph-1))) < 1e-6f);// mulH constant
        // channels 0-6 untouched by fill_consts (still the sentinel)
        CHECK(at(0,0,0) == -123.f);
        CHECK(at(6,3,1) == -123.f);
    }

    // A5: color round-trip — NV12 64x64 (mid-gray + luma gradient) YUV->RGB->YUV.
    // Construct an NV12 aji_frame, run yuv420_to_rgb_planes then rgb_planes_to_yuv420,
    // assert |out - in| <= 2 on Y and chroma over the unpadded window.
    {   const int W=64, H=64;
        auto g = rife_cpu::geometry(W, H);  // 64x64 already mod-64: pw=ph=64, no pad
        const int cw=W>>1, ch=H>>1;
        std::vector<uint8_t> yin((size_t)W*H), uvin((size_t)cw*ch*2);
        // luma: horizontal gradient around mid-gray (clamped to limited-range-ish span)
        for (int y=0;y<H;y++) for (int x=0;x<W;x++)
            yin[(size_t)y*W+x] = (uint8_t)(40 + (x*150)/(W-1));  // 40..190
        // chroma: a gentle 2-D ramp, both Cb and Cr, around 128
        for (int y=0;y<ch;y++) for (int x=0;x<cw;x++) {
            uvin[((size_t)y*cw+x)*2]   = (uint8_t)(112 + (x*30)/(cw-1)); // Cb 112..142
            uvin[((size_t)y*cw+x)*2+1] = (uint8_t)(118 + (y*24)/(ch-1)); // Cr 118..142
        }
        aji_frame in{}; in.width=W; in.height=H; in.format=AJI_FMT_NV12;
        in.matrix=AJI_MATRIX_BT601; in.range=AJI_RANGE_LIMITED; in.siting=AJI_SITING_LEFT;
        in.plane[0]=yin.data(); in.plane[1]=uvin.data(); in.stride[0]=W; in.stride[1]=cw*2;

        const size_t plane=(size_t)g.pw*g.ph;
        std::vector<float> tensor((size_t)11*plane, 0.f);
        rife_cpu::yuv420_to_rgb_planes(in, g, tensor.data(), /*frame_index=*/0, (aji_range)in.range);

        std::vector<uint8_t> yout((size_t)W*H, 0), uvout((size_t)cw*ch*2, 0);
        aji_frame out=in; out.plane[0]=yout.data(); out.plane[1]=uvout.data();
        // RGB tensor is 3-channel-at-offset; rgb_planes_to_yuv420 reads ch 3*0..3*0+2
        rife_cpu::rgb_planes_to_yuv420(tensor.data(), g, out, (aji_range)in.range);

        int maxY=0, maxC=0;
        for (size_t i=0;i<yin.size();i++) maxY = std::max(maxY, std::abs((int)yin[i]-(int)yout[i]));
        for (size_t i=0;i<uvin.size();i++) maxC = std::max(maxC, std::abs((int)uvin[i]-(int)uvout[i]));
        CHECK(maxY <= 2);
        CHECK(maxC <= 2);
        if (maxY>2 || maxC>2) printf("  color round-trip maxY=%d maxC=%d\n", maxY, maxC);
    }

    printf(g_fail ? "rife_selftest: %d FAILURES\n" : "rife_selftest: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
