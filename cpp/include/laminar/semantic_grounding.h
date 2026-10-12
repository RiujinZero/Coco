#pragma once
// Derive compositional language rules from verified, learned procedures.
// No English words, question families or question-specific programs occur here.
#include "procedure_memory.h"
#include <sstream>

namespace laminar::reasoning {
struct Meaning {
    int op = -1, argument = -1; // -1: entity; 4..8: skills; 10/11: membership; 12..15: inverse read; 16..19: unique read
    std::vector<Meaning> children;
    static Meaning var(int i) { return {-1, i, {}}; }
    std::string key() const {
        if (op < 0) return "$" + std::to_string(argument);
        std::string s = std::to_string(op) + "(";
        for (const auto& c : children) s += c.key() + ",";
        return s + ")";
    }
    static Meaning call(int op, std::vector<Meaning> args) {
        // Equality's argument order has no semantic effect. Canonicalize this
        // primitive property so alternative SAVE/SWAP implementations agree.
        if (op == 8 && args.size() == 2 && args[1].key() < args[0].key()) std::swap(args[0], args[1]);
        return {op, -1, std::move(args)};
    }
    int size() const { int n = op >= 0; for (const auto& c : children) n += c.size(); return n; }
    int type() const {
        if (op < 0) return argument >= 0 ? 0 : -1;
        if (op < 4 || op > 19 || op==9 || children.size() != ((op == 8 || op==10 || op==11) ? 2u : 1u)) return -1;
        if(op==10||op==11)return children[0].type()==0&&children[1].type()==0?1:-1;
        if (op == 8) return children[0].type() >= 0 && children[0].type() == children[1].type() ? 1 : -1;
        return children[0].type() == 0 ? (op < 6 || op==12 || op==13 || op>=16 ? 0 : 2) : -1;
    }
    std::set<int> variables() const {
        if (op < 0) return {argument};
        std::set<int> out;
        for (const auto& c : children) { auto v = c.variables(); out.insert(v.begin(), v.end()); }
        return out;
    }
    Meaning rename(const std::map<int, int>& ids) const {
        if (op < 0) return var(ids.at(argument));
        std::vector<Meaning> cs; for (const auto& c : children) cs.push_back(c.rename(ids));
        return call(op, std::move(cs));
    }
    Meaning substitute(const std::vector<Meaning>& args) const {
        if (op < 0) return argument >= 0 && argument < (int)args.size() ? args[argument] : Meaning{};
        std::vector<Meaning> cs; for (const auto& c : children) cs.push_back(c.substitute(args));
        return call(op, std::move(cs));
    }
    void subterms(std::map<std::string, Meaning>& out) const {
        out.emplace(key(), *this); for (const auto& c : children) c.subterms(out);
    }
    static Meaning from_program(const Program& program, int arity) {
        std::array<Meaning, 2> r{var(0), var(arity > 1 ? 1 : 0)};
        for (int a : program) {
            if (a == 9) return r[0];
            if (a == 0) r[0] = var(0);
            else if (a == 1) r[0] = var(arity > 1 ? 1 : 0);
            else if (a == 2) r[1] = r[0];
            else if (a == 3) std::swap(r[0], r[1]);
            else if (a >= 4 && a <= 19 && a!=9) r[0] = call(a, (a == 8 || a==10 || a==11) ? std::vector<Meaning>{r[0], r[1]} : std::vector<Meaning>{r[0]});
            else return {};
        }
        return {};
    }
    Value evaluate(ProcedureMemory& memory, const FlatQuery& q, int depth = 0) const {
        if(depth==0)memory.missing_evidence=memory.nonunique_result=false;
        if (depth > 64 || type() < 0) return {};
        if (op < 0) return argument == 0 ? q.a : argument == 1 ? q.b : Value{};
        Registers args{};
        for (size_t i = 0; i < children.size(); ++i) {
            args[i] = children[i].evaluate(memory, q, depth + 1);
            if (!args[i].present()) return {};
        }
        return memory.step(q, args, op) ? args[0] : Value{};
    }
};

struct SemanticGrounding {
    struct Example { std::string form; Meaning meaning; };
    struct Rule { std::vector<std::string> pattern; Meaning meaning; bool fragment = false; };
    struct Parse { Meaning meaning; size_t states = 0; bool found = false, ambiguous = false, exhausted = false; };
    std::vector<Rule> rules;
    static std::vector<std::string> words(const std::string& s) {
        std::vector<std::string> out; std::istringstream in(s); std::string w;
        while (in >> w) out.push_back(w); return out;
    }
    static int variable(const std::string& w) {
        if (w.size() != 2 || w[0] != '$' || w[1] < '0' || w[1] > '1') return -1;
        return w[1] - '0';
    }
    static std::string join(const std::vector<std::string>& w) {
        std::string s; for (const auto& x : w) s += x + ' '; return s;
    }
    void fit(const std::vector<Example>& examples) {
        rules.clear();
        struct Candidate { std::vector<std::string> pattern; std::map<std::string, Meaning> meanings; std::set<std::string> contexts; bool started = false; };
        std::map<std::string, Candidate> spans;
        std::set<std::string> added;
        auto add = [&](const std::vector<std::string>& pattern, const Meaning& meaning, bool fragment) {
            if (meaning.type() < 0 || pattern.empty() || (pattern.size() == 1 && variable(pattern[0]) >= 0)) return;
            const std::string key = join(pattern) + ":" + meaning.key();
            if (added.insert(key).second) rules.push_back({pattern, meaning, fragment});
        };
        for (const auto& ex : examples) {
            auto ws = words(ex.form); add(ws, ex.meaning, false);
            std::map<std::string, Meaning> subterms; ex.meaning.subterms(subterms);
            for (size_t begin = 0; begin < ws.size(); ++begin) for (size_t end = begin + 1; end <= ws.size(); ++end) {
                std::vector<std::string> pattern(ws.begin() + begin, ws.begin() + end);
                std::map<int, int> rename; bool repeated = false;
                for (auto& w : pattern) if (int v = variable(w); v >= 0) {
                    if (rename.count(v)) { repeated = true; break; }
                    int n = (int)rename.size(); rename[v] = n; w = "$" + std::to_string(n);
                }
                if (rename.empty() || repeated) continue;
                std::set<int> vars; for (auto [old, local] : rename) vars.insert(old);
                std::map<std::string, Meaning> possible;
                for (const auto& [key, tree] : subterms) if (tree.variables() == vars) {
                    Meaning local = tree.rename(rename); possible.emplace(local.key(), std::move(local));
                }
                auto& candidate = spans[join(pattern)]; candidate.pattern = pattern;
                if (!candidate.started) { candidate.meanings = possible; candidate.started = true; }
                else for (auto it = candidate.meanings.begin(); it != candidate.meanings.end();) {
                    if (!possible.count(it->first)) it = candidate.meanings.erase(it); else ++it;
                }
                std::vector<std::string> context(ws.begin(), ws.begin() + begin); context.push_back("<span>");
                context.insert(context.end(), ws.begin() + end, ws.end());
                candidate.contexts.insert(join(context));
            }
        }
        for (const auto& [key, candidate] : spans) {
            if (candidate.contexts.size() < 2) continue;
            // A phrase must have the same grounded meaning in distinct contexts.
            // Retain the largest shared subterm; an entity alone cannot absorb an
            // operation because identity and other contexts provide counterevidence.
            int largest = -1, ties = 0; Meaning best;
            for (const auto& [mk, tree] : candidate.meanings) {
                if (tree.size() > largest) { largest = tree.size(); ties = 1; best = tree; }
                else if (tree.size() == largest) ++ties;
            }
            if (ties == 1) add(candidate.pattern, best, true);
        }
    }
    Parse parse(const std::string& form, bool fragments = true, size_t max_states = 50000) const {
        Parse result; const auto text = words(form); const int n = (int)text.size();
        if (n == 0 || n > 96) { result.exhausted = n > 96; return result; }
        using Cell = std::map<std::string, Meaning>;
        std::vector<std::vector<Cell>> chart(n, std::vector<Cell>(n + 1));
        for (int i = 0; i < n; ++i) if (int v = variable(text[i]); v >= 0) chart[i][i + 1].emplace(text[i], Meaning::var(v));
        for (int length = 2; length <= n; ++length) for (int start = 0; start + length <= n; ++start) {
            const int end = start + length;
            for (const auto& rule : rules) {
                if ((!fragments && rule.fragment) || rule.pattern.size() > (size_t)length) continue;
                if (variable(rule.pattern.front()) < 0 && rule.pattern.front() != text[start]) continue;
                if (variable(rule.pattern.back()) < 0 && rule.pattern.back() != text[end - 1]) continue;
                std::vector<Meaning> args(2);
                auto match = [&](auto&& self, size_t pi, int at) -> void {
                    if (result.exhausted) return;
                    if (++result.states > max_states) { result.exhausted = true; return; }
                    if (pi == rule.pattern.size()) {
                        if (at != end) return;
                        auto meaning = rule.meaning.substitute(args);
                        if (meaning.type() < 0 || meaning.size() > 64) return;
                        auto& cell = chart[start][end]; cell.emplace(meaning.key(), std::move(meaning));
                        if (cell.size() > 16) result.exhausted = true;
                        return;
                    }
                    if (at >= end) return;
                    const int v = variable(rule.pattern[pi]);
                    if (v < 0) { if (text[at] == rule.pattern[pi]) self(self, pi + 1, at + 1); return; }
                    int max_end = end - (int)(rule.pattern.size() - pi - 1);
                    for (int e = at + 1; e <= max_end; ++e) for (const auto& [key, meaning] : chart[at][e]) {
                        args[v] = meaning; self(self, pi + 1, e);
                    }
                };
                match(match, 0, start);
                if (result.exhausted) return result;
            }
        }
        const auto& final = chart[0][n]; result.ambiguous = final.size() > 1;
        result.found = final.size() == 1;
        if (result.found) result.meaning = final.begin()->second;
        return result;
    }
    std::string signature() const {
        std::string out; for (const auto& r : rules) out += join(r.pattern) + ":" + r.meaning.key() + (r.fragment ? "/f\n" : "/q\n"); return out;
    }
};
} // namespace laminar::reasoning
