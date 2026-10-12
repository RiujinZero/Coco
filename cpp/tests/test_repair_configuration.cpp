#include "../apps/continual_checkpoint.h"
#include <iostream>
using namespace laminar;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
    // An actual 131-instruction follow execution exposes the old 128/256
    // acceptance mismatch. This fixture is a contract, not learned evidence.
    struct Follow {
        int choose(const reasoning::Observation& o,const std::vector<int>&) {
            if(!o[4])return 0;
            if(o[10])return 6;
            if(o[6])return 2;
            return 1;
        }
    } follow;
    reasoning::PrimitiveReceipt long_prior;long_prior.cue=0;long_prior.subject={0,0};long_prior.answer={0,64};
    for(int i=0;i<64;++i)long_prior.facts.push_back({i,0,i+1});
    check(!reasoning::verifies_direct_outcome(follow,long_prior,128)&&reasoning::verifies_direct_outcome(follow,long_prior),"long direct outcome rejected by repair acceptance");
    long_prior.answer={0,63};check(!reasoning::verifies_direct_outcome(follow,long_prior),"wrong long outcome accepted");
    auto baseline=training::ContinualTrainingConfig::named("controller50k");
    auto sequential=training::ContinualTrainingConfig::named("controller50k_repair");
    auto interleaved=training::ContinualTrainingConfig::named("controller50k_repair_interleaved");
    auto single_records=training::ContinualTrainingConfig::named("controller50k_repair_records");
    auto bank_records=training::ContinualTrainingConfig::named("controller50k_repair_bank_records");
    check(!baseline.repair_available&&!baseline.interleave_quantum,"baseline changed");
    check(sequential.repair_available&&!sequential.interleave_quantum,"sequential profile changed");
    check(interleaved.repair_available&&interleaved.interleave_quantum==1000&&interleaved.controller_hidden==1316,"experimental profile wrong");
    check(!single_records.repair_bank&&bank_records.repair_bank&&bank_records.repair_available&&bank_records.observation_records,"repair bank profile lost its declared prerequisites");
    check(training::fingerprint(single_records.json(1))!=training::fingerprint(bank_records.json(1)),"repair bank not checkpoint-bound");
    auto bank_controller=bank_records.create(1);
    bool late_bank_rejected=false;
    try{bank_controller.knowledge.enable_composed_learning(1,false,0,true);}catch(const std::invalid_argument&){late_bank_rejected=true;}
    check(late_bank_rejected,"canonical bank configuration did not configure its composed learner");
    auto changed=interleaved;changed.interleave_quantum=500;
    check(training::fingerprint(changed.json(1))!=training::fingerprint(interleaved.json(1)),"quantum not checkpoint-bound");
    reasoning::ComposedLearner learner;learner.configure_interleaving(3);
    reasoning::ComposedReceipt r;r.id="literal";r.question.query.cue=9;
    r.question.query.a={2,7};r.question.query.b={0,4};r.question.query.relation[0]=0;r.question.query.relation[1]=1;r.question.answer={2,7};
    check(learner.teach(r),"fixture rejected");auto observed=learner;
    std::ostringstream progress;observed.search_progress(&progress);
    check(learner.rebuild(1)&&observed.rebuild(1),"literal compile failed");
    check(learner.memory().procedures==observed.memory().procedures&&learner.memory().skills.kernel.routes.W==observed.memory().skills.kernel.routes.W,"observer changed learning");
    check(progress.str().find("compile_begin")!=std::string::npos&&progress.str().find("compile_end")!=std::string::npos,"phase counters missing");
    bool rejected=false;try{learner.configure_interleaving(4);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"late scheduler change accepted");
    std::cout<<"repair configuration fingerprints, immutable schedule and observer parity passed\n";
}
