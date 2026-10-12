// Instantiation check only: the reader component's complete state is reachable by the state archive.
#include "../apps/goal_reader_environment.h"
#include "laminar/state_io.h"
int main() { laminar::goal_learning::reader_task::Component c; laminar::state::Writer w; laminar::state::io(w, c); return (int)(w.fingerprint() & 1); }
