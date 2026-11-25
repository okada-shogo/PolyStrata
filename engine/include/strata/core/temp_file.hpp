// include/strata/core/temp_file.hpp - a file written through a temporary file of its own beside it: the session
// files' way (conversation_file.cpp), for small files written whole (the F32 tile cache, gemm_f32_tiles_autotune()).
// The temporary name is hidden and unique (temp_name()), the file is made only if no file has the name (O_CREAT |
// O_EXCL, never through a link; CreateFileW CREATE_NEW on Windows), and it is renamed over the destination, so two
// processes writing, or one probing while another writes, never share a temporary file and a reader sees the old file
// or the new one whole.
#pragma once

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::core::temp_file {

inline uint64_t random64() {
    constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full;
    std::random_device rd;
    uint64_t r = (uint64_t(rd()) << 32) ^ rd();
    r ^= (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * P1;
#ifdef _WIN32
    r ^= (uint64_t) _getpid() * P2;
#else
    r ^= (uint64_t) ::getpid() * P2;
#endif
    return r;
}

// The temporary name beside the destination: hidden (a leading dot, which the server never accepts as a session
// name), unique, and short enough for any filesystem whatever the destination's length.
inline std::string temp_name(const std::string& leaf) {
    size_t keep = leaf.size() < 64 ? leaf.size() : 64;
    while (keep > 0 && keep < leaf.size() && (uint8_t(leaf[keep]) & 0xC0) == 0x80) --keep;   // whole UTF-8 chars
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) random64());
    return "." + leaf.substr(0, keep) + "." + hex + ".tmp";
}

#ifdef _WIN32
inline bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}
#endif

// A new temporary file beside path holding data; publish: renamed over path, else removed (a writability probe).
// false: path is as it was and no temporary file is left.
inline bool write_beside(const std::string& path, const std::string& data, bool publish) {
#ifdef _WIN32
    const size_t cut = path.find_last_of("/\\");
#else
    const size_t cut = path.find_last_of('/');
#endif
    const std::string dir = cut == std::string::npos ? std::string() : path.substr(0, cut == 0 ? 1 : cut);
    const std::string leaf = cut == std::string::npos ? path : path.substr(cut + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") return false;
#ifdef _WIN32
    std::wstring wdir, wleaf;
    if (!wide(dir, wdir) || !wide(leaf, wleaf)) return false;
    const std::wstring prefix = wdir.empty() ? std::wstring()
                                             : (wdir.back() == L'\\' || wdir.back() == L'/' ? wdir : wdir + L"\\");
    std::wstring temp;
    HANDLE h = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 8 && h == INVALID_HANDLE_VALUE; ++attempt) {
        std::wstring name;
        wide(temp_name(leaf), name);
        temp = prefix + name;
        h = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS)
            return false;
    }
    if (h == INVALID_HANDLE_VALUE) return false;
    bool ok = true;
    for (size_t at = 0; ok && at < data.size();) {
        DWORD n = 0;
        const DWORD want = (DWORD) (data.size() - at < (1u << 30) ? data.size() - at : (1u << 30));
        ok = WriteFile(h, data.data() + at, want, &n, nullptr) && n > 0;
        at += n;
    }
    ok = CloseHandle(h) && ok;
    // MoveFileExW replaces an existing destination (std::rename does not on Windows)
    if (ok && publish && MoveFileExW(temp.c_str(), (prefix + wleaf).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return true;
    DeleteFileW(temp.c_str());   // only the temporary file this call created
    return ok && !publish;
#else
    const int dirfd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirfd < 0) return false;
    std::string temp;
    int fd = -1;
    for (int attempt = 0; attempt < 8 && fd < 0; ++attempt) {
        temp = temp_name(leaf);
        // a new file that did not exist (O_CREAT|O_EXCL: never an existing file or link), readable by its owner only
        fd = ::openat(dirfd, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0 && errno != EEXIST) break;
    }
    if (fd < 0) {
        ::close(dirfd);
        return false;
    }
    bool ok = true;
    for (size_t at = 0; ok && at < data.size();) {
        const ssize_t w = ::write(fd, data.data() + at, data.size() - at);
        if (w < 0 && errno == EINTR) continue;
        ok = w > 0;
        if (ok) at += (size_t) w;
    }
    ok = ::close(fd) == 0 && ok;
    if (ok && publish && ::renameat(dirfd, temp.c_str(), dirfd, leaf.c_str()) == 0) {
        ::close(dirfd);
        return true;
    }
    ::unlinkat(dirfd, temp.c_str(), 0);   // only the temporary file this call created
    ::close(dirfd);
    return ok && !publish;
#endif
}

// data at path, through a temporary file of its own renamed over it
inline bool replace_file(const std::string& path, const std::string& data) { return write_beside(path, data, true); }

// whether a file can be made beside path: a temporary one is made and removed; no existing name is touched
inline bool can_create_beside(const std::string& path) { return write_beside(path, std::string(), false); }

}  // namespace strata::core::temp_file
