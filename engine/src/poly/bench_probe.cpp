// src/poly/bench_probe.cpp - --bench-report's hardware: what the card and the CPU are (gpu_info(), host_info()) and
// what they measure (probe_hardware(), probe_dots()).  The probes run before the model loads, so the card's memory is
// free for their buffers: 256 MiB of pinned memory and 1 GiB of VRAM at most, given back after.
#include "bench_report.hpp"

#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define STRATA_PROBE_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <devpropdef.h>
#include <setupapi.h>   // types only: the functions are loaded at run time (setupapi.dll), nothing is linked
#else
#include <dlfcn.h>
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace strata::poly::bench {
namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\0')) s.pop_back();
    size_t a = 0;
    while (a < s.size() && s[a] == ' ') ++a;
    return s.substr(a);
}

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return trim(s);
}

int pcie_gen_of(const std::string& speed) {   // "8.0 GT/s PCIe" -> 3
    const double gt = std::atof(speed.c_str());
    if (gt <= 0) return -1;
    if (gt < 4) return 1;
    if (gt < 7) return 2;
    if (gt < 12) return 3;
    if (gt < 24) return 4;
    if (gt < 48) return 5;
    return 6;
}

// one timed copy loop on a stream: GB/s of `bytes` a copy.  Timed by the host's clock from the first copy to the
// stream's end, not by events: on Windows (WDDM, HIP) a copy from pinned memory can run on a copy engine that the
// stream's events do not bracket, and 8 copies of 256 MiB read as 32 TB/s.  A few microseconds of launch in 2 GiB of
// copies do not show.
double timed_copies(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps, cudaStream_t s) {
    double gbs = 0.0;
    if (cudaMemcpyAsync(dst, src, bytes, kind, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess) {   // warm
        const Clock::time_point t0 = Clock::now();
        bool ok = true;
        for (int i = 0; i < reps && ok; ++i) ok = cudaMemcpyAsync(dst, src, bytes, kind, s) == cudaSuccess;
        ok = ok && cudaStreamSynchronize(s) == cudaSuccess;
        const double sec = secs(t0);
        if (ok && sec > 0) gbs = (double) bytes * reps / 1e9 / sec;
    }
    (void) cudaGetLastError();
    return gbs;
}

int usable_threads() {
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) return std::max(1, (int) CPU_COUNT(&set));
#elif defined(_WIN32)
    DWORD_PTR mine = 0, all = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &mine, &all) && mine) {
        int n = 0;
        for (DWORD_PTR m = mine; m; m &= m - 1) ++n;
        // (one processor group's mask: past 64 logical CPUs, all the groups' CPUs)
        return n >= 64 ? (int) std::max(64u, std::thread::hardware_concurrency()) : n;
    }
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

// The PCIe link of the card at bus id `bus` ("0000:01:00.0" or CUDA's "00000000:01:00.0"): now (a card in a low-power
// state reports a lower one) and at most.  The bus id is only read here, never written out.
struct Link {
    int gen = -1, width = -1, gen_max = -1, width_max = -1;
    bool any() const { return gen > 0 || width > 0; }
};

#ifdef _WIN32
void* open_lib(const char* name) { return (void*) LoadLibraryA(name); }
void* lib_fn(void* lib, const char* name) { return lib ? (void*) GetProcAddress((HMODULE) lib, name) : nullptr; }
#else
void* open_lib(const char* name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void* lib_fn(void* lib, const char* name) { return lib ? dlsym(lib, name) : nullptr; }
#endif

// NVIDIA, any OS: NVML, loaded at run time (the driver installs it; nothing is linked).  False where it is missing.
bool nvml_link(const std::string& bus, Link& l) {
#if defined(STRATA_USE_HIP)
    (void) bus, (void) l;
    return false;
#else
#ifdef _WIN32
    void* lib = open_lib("nvml.dll");
    if (!lib) {   // older drivers keep it here, off the search path
        char pf[MAX_PATH] = {};
        if (GetEnvironmentVariableA("ProgramW6432", pf, MAX_PATH) || GetEnvironmentVariableA("ProgramFiles", pf, MAX_PATH))
            lib = open_lib((std::string(pf) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll").c_str());
    }
#else
    void* lib = open_lib("libnvidia-ml.so.1");   // (WSL2: /usr/lib/wsl/lib)
#endif
    using Init = int (*)();
    using ByBus = int (*)(const char*, void**);
    using Get = int (*)(void*, unsigned*);
    const auto init = (Init) lib_fn(lib, "nvmlInit_v2");
    const auto stop = (Init) lib_fn(lib, "nvmlShutdown");
    const auto by_bus = (ByBus) lib_fn(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    const Get get[4] = {(Get) lib_fn(lib, "nvmlDeviceGetCurrPcieLinkGeneration"),
                        (Get) lib_fn(lib, "nvmlDeviceGetCurrPcieLinkWidth"),
                        (Get) lib_fn(lib, "nvmlDeviceGetMaxPcieLinkGeneration"),
                        (Get) lib_fn(lib, "nvmlDeviceGetMaxPcieLinkWidth")};
    if (!init || !stop || !by_bus || init() != 0) return false;   // (the library stays loaded: the process is short)
    void* dev = nullptr;
    if (by_bus(bus.c_str(), &dev) == 0 && dev) {
        int* out[4] = {&l.gen, &l.width, &l.gen_max, &l.width_max};
        for (int k = 0; k < 4; ++k) {
            unsigned v = 0;
            if (get[k] && get[k](dev, &v) == 0 && v > 0 && v < 64) *out[k] = (int) v;
        }
    }
    stop();
    return l.any();
#endif
}

// Windows, any vendor: the PCI device's link properties through SetupAPI (loaded at run time), found by bus number,
// device and function among the display adapters.
bool setupapi_link(const std::string& bus, Link& l) {
#ifdef _WIN32
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus.c_str(), "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return false;
    void* lib = open_lib("setupapi.dll");
    using ClassDevs = HDEVINFO(WINAPI*)(const GUID*, PCWSTR, HWND, DWORD);
    using Enum = BOOL(WINAPI*)(HDEVINFO, DWORD, PSP_DEVINFO_DATA);
    using RegProp = BOOL(WINAPI*)(HDEVINFO, PSP_DEVINFO_DATA, DWORD, PDWORD, PBYTE, DWORD, PDWORD);
    using DevProp = BOOL(WINAPI*)(HDEVINFO, PSP_DEVINFO_DATA, const DEVPROPKEY*, DEVPROPTYPE*, PBYTE, DWORD, PDWORD, DWORD);
    using Destroy = BOOL(WINAPI*)(HDEVINFO);
    const auto class_devs = (ClassDevs) lib_fn(lib, "SetupDiGetClassDevsW");
    const auto enum_dev = (Enum) lib_fn(lib, "SetupDiEnumDeviceInfo");
    const auto reg_prop = (RegProp) lib_fn(lib, "SetupDiGetDeviceRegistryPropertyW");
    const auto dev_prop = (DevProp) lib_fn(lib, "SetupDiGetDevicePropertyW");
    const auto destroy = (Destroy) lib_fn(lib, "SetupDiDestroyDeviceInfoList");
    if (!class_devs || !enum_dev || !reg_prop || !dev_prop || !destroy) return false;
    static const GUID display = {0x4d36e968, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};
    // DEVPKEY_PciDevice_CurrentLinkSpeed (9), CurrentLinkWidth (10), MaxLinkSpeed (11), MaxLinkWidth (12) of pciprop.h
    static const GUID pci = {0x3ab22e31, 0x8264, 0x4b4e, {0x9a, 0xf5, 0xa8, 0xd2, 0xd8, 0xe3, 0x3e, 0x62}};
    const HDEVINFO set = class_devs(&display, nullptr, nullptr, DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) return false;
    SP_DEVINFO_DATA info{};
    info.cbSize = sizeof info;
    for (DWORD i = 0; enum_dev(set, i, &info); ++i) {
        DWORD bus_no = 0, addr = 0, type = 0;
        if (!reg_prop(set, &info, SPDRP_BUSNUMBER, &type, (PBYTE) &bus_no, sizeof bus_no, nullptr) ||
            !reg_prop(set, &info, SPDRP_ADDRESS, &type, (PBYTE) &addr, sizeof addr, nullptr))
            continue;
        if (bus_no != b || (addr >> 16) != d || (addr & 0xffff) != f) continue;
        int* out[4] = {&l.gen, &l.width, &l.gen_max, &l.width_max};
        const ULONG pid[4] = {9, 10, 11, 12};
        for (int k = 0; k < 4; ++k) {
            DEVPROPKEY key{pci, pid[k]};
            DEVPROPTYPE t = 0;
            UINT32 v = 0;
            if (dev_prop(set, &info, &key, &t, (PBYTE) &v, sizeof v, nullptr, 0) && v > 0 && v < 64)
                *out[k] = (int) v;   // the speeds are 1 = 2.5 GT/s, 2 = 5, 3 = 8, 4 = 16, 5 = 32: the generation
        }
        break;
    }
    destroy(set);
    return l.any();
#else
    (void) bus, (void) l;
    return false;
#endif
}

// Linux: sysfs, the kernel's own view of the link.
bool sysfs_link(std::string id, Link& l) {
#if defined(__linux__)
    if (id.size() == 16) id = id.substr(4);   // 00000000:01:00.0 (CUDA's) -> 0000:01:00.0 (sysfs')
    const std::string dir = "/sys/bus/pci/devices/" + id + "/";
    l.gen = pcie_gen_of(read_line(dir + "current_link_speed"));
    l.gen_max = pcie_gen_of(read_line(dir + "max_link_speed"));
    const std::string w = read_line(dir + "current_link_width"), wm = read_line(dir + "max_link_width");
    l.width = w.empty() ? -1 : std::max(-1, std::atoi(w.c_str()));
    l.width_max = wm.empty() ? -1 : std::max(-1, std::atoi(wm.c_str()));
    if (l.width == 0) l.width = -1;
    if (l.width_max == 0) l.width_max = -1;
    return l.any();
#else
    (void) id, (void) l;
    return false;
#endif
}

}  // namespace

Gpu gpu_info(int device) {
    Gpu g;
#if defined(STRATA_USE_HIP)
    g.backend = "hip";
#else
    g.backend = "cuda";
#endif
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) {
        (void) cudaGetLastError();
        g.name = "unknown";
        return g;
    }
    g.name = trim(p.name);
    g.sms = p.multiProcessorCount;
    g.vram_gib = (double) p.totalGlobalMem / (double) (1ull << 30);
#if defined(STRATA_USE_HIP)
    g.arch = trim(p.gcnArchName);
    const size_t colon = g.arch.find(':');   // gfx1201:sramecc-:xnack- -> gfx1201
    if (colon != std::string::npos) g.arch.resize(colon);
#else
    g.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
#endif
    int v = 0;
    if (cudaDriverGetVersion(&v) == cudaSuccess) {
#if defined(STRATA_USE_HIP)
        g.driver = "HIP driver " + std::to_string(v);
#else
        g.driver = "CUDA " + std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10);
#endif
    }
    if (cudaRuntimeGetVersion(&v) == cudaSuccess) {
#if defined(STRATA_USE_HIP)
        g.runtime = "HIP " + std::to_string(v / 10000000) + "." + std::to_string(v / 100000 % 100);
#else
        g.runtime = "CUDA " + std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10);
#endif
    }
    int clock_khz = 0, bus_bits = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device) == cudaSuccess && clock_khz > 0)
        g.mem_peak_gbs = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;   // double data rate
    (void) cudaGetLastError();
    char bus[32] = {};
#if defined(STRATA_USE_HIP)
    const bool have_bus = hipDeviceGetPCIBusId(bus, (int) sizeof bus, device) == hipSuccess;
#else
    const bool have_bus = cudaDeviceGetPCIBusId(bus, (int) sizeof bus, device) == cudaSuccess;
#endif
    (void) cudaGetLastError();
    if (have_bus) {
        std::string id = bus;
        for (char& c : id) c = (char) std::tolower((unsigned char) c);
        // NVML (NVIDIA, any OS), then SetupAPI (Windows, any vendor), then sysfs (Linux)
        Link l;
        if (nvml_link(id, l)) g.pcie_source = "nvml";
        else if (setupapi_link(id, l = Link{})) g.pcie_source = "setupapi";
        else if (sysfs_link(id, l = Link{})) g.pcie_source = "sysfs";
        g.pcie_gen = l.gen, g.pcie_width = l.width, g.pcie_gen_max = l.gen_max, g.pcie_width_max = l.width_max;
    }
    return g;
}

Host host_info() {
    Host h;
#if defined(STRATA_PROBE_X86)
    unsigned r[12] = {};
    bool brand = false;
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000);
    if ((unsigned) x[0] >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) __cpuid((int*) (r + 4 * i), (int) (0x80000002 + i));
        brand = true;
    }
#else
    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004u) {
        for (unsigned i = 0; i < 3; ++i) __get_cpuid(0x80000002 + i, r + 4 * i, r + 4 * i + 1, r + 4 * i + 2, r + 4 * i + 3);
        brand = true;
    }
#endif
    if (brand) {
        char b[49] = {};
        std::memcpy(b, r, 48);
        h.cpu = trim(b);
    }
#endif
#if defined(__linux__)
    if (h.cpu.empty()) {   // arm64: the first "model name" or "Hardware" line
        std::ifstream f("/proc/cpuinfo");
        for (std::string line; std::getline(f, line);)
            if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
                const size_t c = line.find(':');
                if (c != std::string::npos) h.cpu = trim(line.substr(c + 1));
                break;
            }
    }
#endif
    if (h.cpu.empty()) h.cpu = "unknown";
    // repeated spaces in brand strings ("Intel(R) Core(TM) i7    CPU")
    std::string c;
    for (char ch : h.cpu)
        if (!(ch == ' ' && !c.empty() && c.back() == ' ')) c += ch;
    h.cpu = c;
    namespace Q = strata::quant;
    const Q::Caps& k = Q::caps();
    h.isa = Q::isa_name(Q::best_isa());
    const std::pair<bool, const char*> flags[] = {{k.sse42, "sse4.2"},      {k.avx, "avx"},
                                                  {k.avx2, "avx2"},         {k.bmi2, "bmi2"},
                                                  {k.avx_vnni, "avx_vnni"}, {k.avx512, "avx512"},
                                                  {k.avx512_vnni, "avx512_vnni"}, {k.avx512_vbmi, "avx512_vbmi"},
                                                  {k.neon, "neon"},         {k.dotprod, "dotprod"},
                                                  {k.i8mm, "i8mm"}};
    for (const auto& [on, name] : flags)
        if (on) h.flags.push_back(name);
    h.threads = usable_threads();
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) h.ram_gib = (double) ms.ullTotalPhys / (double) (1ull << 30);
    h.os = "Windows";
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) h.ram_gib = (double) pages * (double) page / (double) (1ull << 30);
    utsname u{};
    if (uname(&u) == 0) h.os = std::string(u.sysname) + " " + u.release;   // the kernel's release, not the host name
#endif
    return h;
}

void probe_hardware(Probes& p, int device, void (*log)(const char*)) {
    const Clock::time_point t0 = Clock::now();
    const auto say = [&](const char* what) {
        if (log) log(what);
    };
    (void) cudaSetDevice(device);
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    const size_t mb = 256ull << 20, big = 512ull << 20;
    // a card that something else is using: its probes would measure that too, and take its memory
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) (void) cudaGetLastError();
    const bool room = free_b >= 2 * big + (512ull << 20);
    if (!room) say("skipped the copies: under 1.5 GiB of the card's memory is free (another program is using it)");
    void* pin = nullptr;
    void* dev = nullptr;
    void* dev2 = nullptr;
    if (room && cudaMallocHost(&pin, mb) == cudaSuccess && cudaMalloc(&dev, mb) == cudaSuccess) {
        std::memset(pin, 1, mb);
        p.h2d_pinned_gbs = timed_copies(dev, pin, mb, cudaMemcpyHostToDevice, 8, s);
        say("probe H2D pinned done");
        p.d2h_pinned_gbs = timed_copies(pin, dev, mb, cudaMemcpyDeviceToHost, 8, s);
        say("probe D2H pinned done");
        std::unique_ptr<uint8_t[]> pageable(new (std::nothrow) uint8_t[mb]);
        if (pageable) {
            std::memset(pageable.get(), 1, mb);
            p.h2d_pageable_gbs = timed_copies(dev, pageable.get(), mb, cudaMemcpyHostToDevice, 4, s);
            say("probe H2D pageable done");
        }
    }
    (void) cudaGetLastError();
    if (dev) (void) cudaFree(dev);
    if (pin) (void) cudaFreeHost(pin);
    dev = nullptr;
    // the card's memory: device-to-device copies of 512 MiB, each a read and a write of it
    if (room && cudaMalloc(&dev, big) == cudaSuccess && cudaMalloc(&dev2, big) == cudaSuccess) {
        (void) cudaMemsetAsync(dev, 1, big, s);
        p.gpu_mem_gbs = 2.0 * timed_copies(dev2, dev, big, cudaMemcpyDeviceToDevice, 10, s);
        say("probe GPU memory done");
    }
    (void) cudaGetLastError();
    if (dev) (void) cudaFree(dev);
    if (dev2) (void) cudaFree(dev2);
    (void) cudaStreamDestroy(s);
    (void) cudaGetLastError();

    // RAM: every usable thread sums its part of 1 GiB (a sixteenth of the RAM at most), best of three passes
    const int threads = usable_threads();
    size_t ram = 1ull << 30;
#ifndef _WIN32
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) ram = std::min(ram, (size_t) pages * (size_t) page / 16);
#endif
    const size_t n = ram / sizeof(uint64_t);
    std::unique_ptr<uint64_t[]> buf(new (std::nothrow) uint64_t[n]);
    if (buf) {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)   // touched by the threads that read it: on a NUMA PC, each on its node
            th.emplace_back([&, t] {
                const size_t a = n * (size_t) t / (size_t) threads, b = n * (size_t) (t + 1) / (size_t) threads;
                for (size_t i = a; i < b; ++i) buf[i] = i;
            });
        for (std::thread& x : th) x.join();
        double best = 0.0;
        std::atomic<uint64_t> sink{0};
        for (int pass = 0; pass < 3; ++pass) {
            th.clear();
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            Clock::time_point start;
            for (int t = 0; t < threads; ++t)
                th.emplace_back([&, t] {
                    const size_t a = n * (size_t) t / (size_t) threads, b = n * (size_t) (t + 1) / (size_t) threads;
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    size_t i = a;
                    for (; i + 4 <= b; i += 4) s0 += buf[i], s1 += buf[i + 1], s2 += buf[i + 2], s3 += buf[i + 3];
                    for (; i < b; ++i) s0 += buf[i];
                    sink.fetch_add(s0 + s1 + s2 + s3, std::memory_order_relaxed);
                });
            while (ready.load() < threads) std::this_thread::yield();
            start = Clock::now();
            go.store(true, std::memory_order_release);
            for (std::thread& x : th) x.join();
            const double sec = secs(start);
            if (sec > 0) best = std::max(best, (double) ram / 1e9 / sec);
        }
        p.ram_read_gbs = best;
        p.ram_threads = threads;
        say("probe RAM read done");
    }
    p.seconds += secs(t0);
}

void probe_dots(Probes& p, const std::vector<std::pair<int, int>>& type_cols, int threads) {
    namespace Q = strata::quant;
    const Clock::time_point t0 = Clock::now();
    threads = std::max(threads, 1);
    std::vector<int> seen;
    for (const auto& [type, cols] : type_cols) {
        if (std::find(seen.begin(), seen.end(), type) != seen.end()) continue;   // a rate a type
        seen.push_back(type);
        const Q::TypeInfo* ti = Q::type_info(type);
        const Q::DotKernel* k = ti ? Q::dot_kernel(ti->type, Q::default_arith()) : nullptr;
        // the activation format the kernel takes (DotKernel::act), which is the one cpu_experts() quantizes to
        const size_t rb = Q::row_bytes(type, cols), ab = k ? Q::row_bytes((int) k->act, cols) : 0;
        if (!k || !k->fn || rb == 0 || ab == 0) continue;
        // 512 MiB of rows: far more than any cache, as the experts outside VRAM are
        const size_t rows = std::max<size_t>((512ull << 20) / rb, (size_t) threads);
        std::unique_ptr<uint8_t[]> w(new (std::nothrow) uint8_t[rows * rb]);
        if (!w) continue;
        std::vector<uint8_t> act(ab, 0);
        // small, finite scales in every block (0x11 bytes: an fp16 of 2^-11 and so on); the values do not change the time
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&, t] {
                const size_t a = rows * (size_t) t / (size_t) threads, b = rows * (size_t) (t + 1) / (size_t) threads;
                std::memset(w.get() + a * rb, 0x11, (b - a) * rb);
            });
        for (std::thread& x : th) x.join();
        const auto pass = [&](int nt, size_t nrows) {
            std::vector<std::thread> ts;
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::atomic<uint32_t> sink{0};
            for (int t = 0; t < nt; ++t)
                ts.emplace_back([&, t] {
                    const size_t a = nrows * (size_t) t / (size_t) nt, b = nrows * (size_t) (t + 1) / (size_t) nt;
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                    float acc = 0.f, s = 0.f;
                    for (size_t r = a; r < b; ++r) {
                        k->fn(cols, &s, w.get() + r * rb, act.data());
                        acc += s;
                    }
                    uint32_t bits;
                    std::memcpy(&bits, &acc, 4);
                    sink.fetch_add(bits, std::memory_order_relaxed);
                });
            while (ready.load() < nt) std::this_thread::yield();
            const Clock::time_point start = Clock::now();
            go.store(true, std::memory_order_release);
            for (std::thread& x : ts) x.join();
            const double sec = secs(start);
            return sec > 0 ? (double) nrows * (double) rb / 1e9 / sec : 0.0;
        };
        DotRate d;
        d.type = ti->name;
        d.kernel = k->name;
        d.isa = Q::isa_name(k->isa);
        d.threads = threads;
        pass(threads, rows);   // warm
        d.gbs_all = std::max(pass(threads, rows), pass(threads, rows));
        d.gbs_one = pass(1, std::min(rows, (size_t) ((64ull << 20) / rb) + 1));
        p.dots.push_back(d);
    }
    p.seconds += secs(t0);
}

}  // namespace strata::poly::bench
