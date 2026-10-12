// Loads a subsystem state file (subsystem_run PREFIX_final.state or _checkpoint.state) into a fresh controller and
// reader, verifies the trailing hash, and prints the state fingerprint (compare with PREFIX_summary.json).
// usage: subsystem_state_check FILE
#include "goal_extended_declaration.h"
#include "laminar/state_io.h"
struct RunState {   // same layout as subsystem_run.cpp
    uint64_t next = 0, fits = 0, warm_success = 0, autonomous_success = 0;
    std::array<uint64_t, 8> occurrences{}, successes{};
    std::vector<uint64_t> log_sizes;
    template <class Ar> void state_io(Ar& ar) { ar(next); ar(fits); ar(warm_success); ar(autonomous_success); ar(occurrences); ar(successes); ar(log_sizes); }
};
int main(int argc, char** argv) {
    try {
        gl::require(argc == 2, "usage: subsystem_state_check FILE");
        GoalControlledRelationalSession learner(1, 1316, true); rt::Component reader;
        RunState rs;
        { laminar::state::Reader r(argv[1]); r.section("subsystem"); laminar::state::io(r, learner); r.section("reader"); laminar::state::io(r, reader); r.section("run"); laminar::state::io(r, rs); r.finish(); }
        laminar::state::Writer w; laminar::state::io(w, learner); laminar::state::io(w, reader);
        std::cout << "loaded " << argv[1] << ": missions " << rs.next << ", updates " << learner.updates() << ", credits " << learner.credits().size()
                  << ", knowledge version " << learner.knowledge.version() << ", fingerprint " << std::hex << w.fingerprint() << std::dec << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
