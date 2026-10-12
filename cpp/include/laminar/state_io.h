#pragma once
// Binary state archive for long-running training. One generic io(Ar&, T&) serves both directions, so the
// saved and loaded member lists cannot drift apart. Every section is labelled and length-checked; the file
// ends with a hash of everything written. A Writer without a file computes only the hash: that is the state
// fingerprint used by resume tests.
//
// Dispatch, in order: a member state_io(Ar&); std::string; std::vector; std::array; associative containers;
// optional / pair / tuple / variant / deque / list; owned shared_ptr / unique_ptr (the pointee is saved by value,
// so aliasing between separate owners is NOT preserved); trivially copyable values as raw bytes; and finally
// plain aggregates member by member through structured bindings (state_reflect_gen.inc). Raw pointers and
// std::function are refused at compile time: a type holding them needs its own state_io.
//
// Payloads of 4 KiB or more are stored in 4 KiB chunks, each tagged as all-zero (no data follows) or raw.
// A model's weight and plasticity buffers are sized for its full unit capacity but mostly unallocated, so
// this keeps checkpoints proportional to the live population. The hash always covers the logical bytes, so
// fingerprints do not depend on the encoding. finish() flushes and fsyncs before the caller renames the file.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#if defined(__unix__)
#include <unistd.h>
#endif

namespace laminar::state {

inline constexpr size_t sparse_min = 4096, sparse_chunk = 4096;
inline bool all_zero(const uint8_t* p, size_t n) {
    size_t i = 0; uint64_t acc = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; std::memcpy(&w, p + i, 8); acc |= w; if (acc) return false; }
    for (; i < n; ++i) acc |= p[i];
    return acc == 0;
}

struct Hash {
    uint64_t h = 0x243F6A8885A308D3ULL, n = 0;
    void add(const void* p, size_t len) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        size_t i = 0;
        for (; i + 8 <= len; i += 8) { uint64_t w; std::memcpy(&w, b + i, 8); mix(w); }
        uint64_t tail = 0; if (len > i) std::memcpy(&tail, b + i, len - i);   // an empty vector's data() may be null
        mix(tail ^ ((uint64_t)(len - i) << 56));
        n += len;
    }
    void mix(uint64_t w) { h ^= w + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 32; }
    uint64_t value() const { uint64_t x = h ^ n; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL; x ^= x >> 33; return x; }
};

// ---- type traits and aggregate reflection -------------------------------------------------------------------
namespace detail {
template <class T, template <class...> class Tpl> struct is_spec : std::false_type {};
template <template <class...> class Tpl, class... A> struct is_spec<Tpl<A...>, Tpl> : std::true_type {};
template <class T> struct is_std_array : std::false_type {};
template <class T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};
template <class T> inline constexpr bool always_false = false;
template <class Ar, class T> concept has_state_io = requires(Ar& ar, T& v) { v.state_io(ar); };

struct any_member { template <class T> operator T() const; };   // unevaluated only
template <class T, size_t... I> constexpr bool brace_constructible(std::index_sequence<I...>) {
    return requires { T{((void)I, any_member{})...}; };
}
template <class T, size_t N = 0> constexpr size_t member_count() {
    if constexpr (N > 64) return N;
    else if constexpr (!brace_constructible<T>(std::make_index_sequence<N + 1>{})) return N;
    else return member_count<T, N + 1>();
}
#include "state_reflect_gen.inc"
} // namespace detail

template <class Ar, class T> void io(Ar& ar, T& v);

namespace detail {
template <class Variant, class Ar, size_t... I> void variant_emplace(Variant& v, size_t index, Ar& ar, std::index_sequence<I...>) {
    bool done = false;
    ((index == I ? (v.template emplace<I>(), io(ar, std::get<I>(v)), done = true) : false), ...);
    if (!done) throw std::runtime_error("state variant index out of range");
}
template <class Ar, class C> void sequence_io(Ar& ar, C& c) {
    uint64_t n = c.size(); ar.bytes(&n, 8);
    if constexpr (Ar::loading) {
        if (n > (1ULL << 36)) throw std::runtime_error("state container length implausible");
        c.clear(); for (uint64_t i = 0; i < n; ++i) { typename C::value_type x{}; io(ar, x); c.push_back(std::move(x)); }
    } else for (auto& x : c) io(ar, x);
}
template <class Ar, class C> void map_io(Ar& ar, C& c) {
    uint64_t n = c.size(); ar.bytes(&n, 8);
    if constexpr (Ar::loading) {
        if (n > (1ULL << 36)) throw std::runtime_error("state container length implausible");
        c.clear();
        for (uint64_t i = 0; i < n; ++i) { typename C::key_type k{}; typename C::mapped_type m{}; io(ar, k); io(ar, m); c.emplace(std::move(k), std::move(m)); }
    } else for (auto& [k, m] : c) { io(ar, const_cast<typename C::key_type&>(k)); io(ar, m); }
}
template <class Ar, class C> void set_io(Ar& ar, C& c) {
    uint64_t n = c.size(); ar.bytes(&n, 8);
    if constexpr (Ar::loading) {
        if (n > (1ULL << 36)) throw std::runtime_error("state container length implausible");
        c.clear(); for (uint64_t i = 0; i < n; ++i) { typename C::key_type k{}; io(ar, k); c.insert(std::move(k)); }
    } else for (auto& k : c) io(ar, const_cast<typename C::key_type&>(k));
}
} // namespace detail

template <class Ar, class T> void io(Ar& ar, T& v) {
    using U = std::remove_cv_t<T>;
    if constexpr (detail::has_state_io<Ar, U>) v.state_io(ar);
    else if constexpr (std::is_pointer_v<U> || std::is_member_pointer_v<U>) static_assert(detail::always_false<U>, "raw pointers cannot be checkpointed: give the owning type a state_io");
    else if constexpr (detail::is_spec<U, std::function>::value) static_assert(detail::always_false<U>, "std::function cannot be checkpointed: give the owning type a state_io");
    else if constexpr (std::is_same_v<U, std::string>) {
        uint64_t n = v.size(); ar.bytes(&n, 8);
        if constexpr (Ar::loading) { if (n > (1ULL << 32)) throw std::runtime_error("state string implausible"); v.resize(n); }
        ar.bytes(v.data(), n);
    }
    else if constexpr (std::is_same_v<U, std::vector<bool>>) {
        uint64_t n = v.size(); ar.bytes(&n, 8);
        std::vector<uint8_t> b(v.begin(), v.end()); if constexpr (Ar::loading) b.resize(n);
        ar.bytes(b.data(), b.size());
        if constexpr (Ar::loading) v.assign(b.begin(), b.end());
    }
    else if constexpr (detail::is_spec<U, std::vector>::value) {
        uint64_t n = v.size(); ar.bytes(&n, 8);
        if constexpr (Ar::loading) { if (n > (1ULL << 36)) throw std::runtime_error("state vector length implausible"); v.resize(n); }
        if constexpr (std::is_trivially_copyable_v<typename U::value_type> && !std::is_pointer_v<typename U::value_type>) ar.bytes(v.data(), n * sizeof(typename U::value_type));
        else for (auto& x : v) io(ar, x);
    }
    else if constexpr (detail::is_std_array<U>::value && !std::is_trivially_copyable_v<U>) { for (auto& x : v) io(ar, x); }
    else if constexpr (detail::is_spec<U, std::map>::value || detail::is_spec<U, std::unordered_map>::value || detail::is_spec<U, std::multimap>::value) detail::map_io(ar, v);
    else if constexpr (detail::is_spec<U, std::set>::value || detail::is_spec<U, std::unordered_set>::value || detail::is_spec<U, std::multiset>::value) detail::set_io(ar, v);
    else if constexpr (detail::is_spec<U, std::deque>::value || detail::is_spec<U, std::list>::value) detail::sequence_io(ar, v);
    else if constexpr (detail::is_spec<U, std::optional>::value) {
        uint8_t has = v.has_value(); ar.bytes(&has, 1);
        if constexpr (Ar::loading) { if (has) { v.emplace(); io(ar, *v); } else v.reset(); }
        else if (has) io(ar, *v);
    }
    else if constexpr (detail::is_spec<U, std::pair>::value) { io(ar, const_cast<std::remove_const_t<typename U::first_type>&>(v.first)); io(ar, v.second); }
    else if constexpr (detail::is_spec<U, std::tuple>::value) std::apply([&](auto&... x) { (io(ar, x), ...); }, v);
    else if constexpr (detail::is_spec<U, std::variant>::value) {
        uint64_t index = v.index(); ar.bytes(&index, 8);
        if constexpr (Ar::loading) detail::variant_emplace(v, (size_t)index, ar, std::make_index_sequence<std::variant_size_v<U>>{});
        else std::visit([&](auto& x) { io(ar, x); }, v);
    }
    else if constexpr (detail::is_spec<U, std::shared_ptr>::value || detail::is_spec<U, std::unique_ptr>::value) {
        using E = std::remove_const_t<typename U::element_type>;
        uint8_t has = v != nullptr; ar.bytes(&has, 1);
        if constexpr (Ar::loading) {
            if (has) {
                if constexpr (detail::is_spec<U, std::shared_ptr>::value) { auto p = std::make_shared<E>(); io(ar, *p); v = std::move(p); }
                else { auto p = std::make_unique<E>(); io(ar, *p); v = std::move(p); }
            } else v.reset();
        } else if (has) io(ar, const_cast<E&>(*v));
    }
    else if constexpr (std::is_trivially_copyable_v<U>) ar.bytes(&v, sizeof(U));
    else if constexpr (std::is_aggregate_v<U>) detail::for_each_member(v, [&](auto& m) { io(ar, m); });
    else static_assert(detail::always_false<U>, "type needs a state_io member to be checkpointed");
}

class Writer {
    std::FILE* f_ = nullptr;
    Hash hash_;
    void put(const void* p, size_t n) { if (f_ && n && std::fwrite(p, 1, n, f_) != n) throw std::runtime_error("state write failed"); }
public:
    static constexpr bool loading = false;
    Writer() = default;                                   // hash-only fingerprint
    explicit Writer(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {
        if (!f_) throw std::runtime_error("cannot open state file for writing: " + path);
        static const char magic[8] = {'C', 'O', 'C', 'O', 'S', 'T', '0', '2'}; bytes(magic, 8);
    }
    ~Writer() { if (f_) std::fclose(f_); }
    Writer(const Writer&) = delete; Writer& operator=(const Writer&) = delete;
    void bytes(const void* p, size_t n) {
        hash_.add(p, n);
        if (!f_) return;
        if (n < sparse_min) { put(p, n); return; }
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t o = 0; o < n; o += sparse_chunk) {
            const size_t c = std::min(sparse_chunk, n - o);
            const uint8_t tag = all_zero(b + o, c) ? 0 : 1; put(&tag, 1);
            if (tag) put(b + o, c);
        }
    }
    void section(const char* name) { uint32_t n = (uint32_t)std::strlen(name); bytes(&n, 4); bytes(name, n); }
    template <class T> void operator()(T& v) { io(*this, v); }
    template <class T> void array(T* p, size_t n) { static_assert(std::is_trivially_copyable_v<T>); bytes(p, n * sizeof(T)); }
    uint64_t fingerprint() const { return hash_.value(); }
    uint64_t size() const { return hash_.n; }
    // Appends the hash of everything written, flushes, fsyncs and closes the file.
    void finish() {
        uint64_t h = hash_.value();
        if (f_) {
            if (std::fwrite(&h, 8, 1, f_) != 1 || std::fflush(f_) != 0) throw std::runtime_error("state finish failed");
#if defined(__unix__)
            if (::fsync(fileno(f_)) != 0) throw std::runtime_error("state fsync failed");   // durable before any rename
#endif
            if (std::fclose(f_) != 0) { f_ = nullptr; throw std::runtime_error("state close failed"); }
            f_ = nullptr;
        }
    }
};

class Reader {
    std::FILE* f_ = nullptr;
    Hash hash_;
    void get(void* p, size_t n) { if (n && std::fread(p, 1, n, f_) != n) throw std::runtime_error("state file truncated"); }
public:
    static constexpr bool loading = true;
    explicit Reader(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
        if (!f_) throw std::runtime_error("cannot open state file: " + path);
        char magic[8]; bytes(magic, 8);
        if (std::memcmp(magic, "COCOST02", 8) != 0) throw std::runtime_error("not a Coco state file (or an older format): " + path);
    }
    ~Reader() { if (f_) std::fclose(f_); }
    Reader(const Reader&) = delete; Reader& operator=(const Reader&) = delete;
    void bytes(void* p, size_t n) {
        if (n < sparse_min) get(p, n);
        else {
            uint8_t* b = static_cast<uint8_t*>(p);
            for (size_t o = 0; o < n; o += sparse_chunk) {
                const size_t c = std::min(sparse_chunk, n - o); uint8_t tag = 2; get(&tag, 1);
                if (tag == 0) std::memset(b + o, 0, c); else if (tag == 1) get(b + o, c); else throw std::runtime_error("state chunk tag corrupt");
            }
        }
        hash_.add(p, n);
    }
    void section(const char* name) {
        uint32_t n = 0; bytes(&n, 4);
        if (n != std::strlen(name) || n > 256) throw std::runtime_error(std::string("state section mismatch, expected ") + name);
        std::string got(n, '\0'); bytes(got.data(), n);
        if (got != name) throw std::runtime_error("state section mismatch: expected " + std::string(name) + ", found " + got);
    }
    template <class T> void operator()(T& v) { io(*this, v); }
    template <class T> void array(T* p, size_t n) { static_assert(std::is_trivially_copyable_v<T>); bytes(p, n * sizeof(T)); }
    // Verifies the trailing hash; a mismatch means a corrupt or partially written file.
    void finish() {
        const uint64_t expected = hash_.value(); uint64_t stored = 0;
        if (std::fread(&stored, 8, 1, f_) != 1) throw std::runtime_error("state file missing its hash");
        if (stored != expected) throw std::runtime_error("state file hash mismatch");
        if (std::fgetc(f_) != EOF) throw std::runtime_error("trailing bytes after state hash");
    }
};

} // namespace laminar::state
