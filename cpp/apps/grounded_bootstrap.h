#pragma once
// Load teaching inputs for the optional grounded session. This adapter contains
// no world templates, English meaning tables or held-out answer readers.
// Every side-effecting call is checked at runtime, including NDEBUG builds.
#include "laminar/grounded_session.h"
#include "laminar/bootstrap_grounding.h"
#include "laminar/incremental_grounding.h"
#include "laminar/observation_language_learner.h"
#include <filesystem>
#include <fstream>
#include <random>

namespace laminar::reasoning {
struct GroundedBootstrap {
    static void require(bool okay,const std::string& message) {if(!okay)throw std::runtime_error("grounded bootstrap: "+message);}
    static std::ifstream open(const std::string& path) {std::ifstream in(path);require(bool(in),"cannot open "+path);return in;}
    static std::string line(std::istream& in) {
        std::string text;require(bool(std::getline(in,text)),"truncated input");if(!text.empty()&&text.back()=='\r')text.pop_back();return text;
    }
    static std::vector<std::string> fields(const std::string& text) {
        std::vector<std::string> out;std::istringstream in(text);std::string value;while(std::getline(in,value,'\t'))out.push_back(value);return out;
    }
    static void feed(LanguageReasoner& r,const std::string& text) {for(uint8_t c:text)r.observe(c,false);}
    static std::vector<BootstrapGrounding::Episode> lessons(const std::string& path) {
        auto in=open(path);std::vector<BootstrapGrounding::Episode> out;
        while(in.peek()!=EOF) {
            auto h=fields(line(in));require(h.size()==2,"invalid QA header");BootstrapGrounding::Episode e;e.id=std::stoull(h[0]);
            int n=std::stoi(h[1]);require(n>=0&&n<=4096,"invalid QA report count");
            for(int i=0;i<n;++i)e.observations.push_back(line(in));e.question=line(in);e.answer=line(in);out.push_back(std::move(e));
        }
        require(!out.empty(),"empty QA corpus");return out;
    }
    static void primitive(LanguageReasoner& r,const std::string& directory,int seed) {
        auto in=open((std::filesystem::path(directory)/("micro_"+std::to_string(seed)+".txt")).string());
        std::vector<Sample> samples;Sample s;int count;
        while(in>>s.action) {
            require(bool(in>>count)&&count>0&&count<=64,"invalid primitive action menu");s.available.resize(count);
            for(int& a:s.available)require(bool(in>>a),"truncated primitive actions");
            for(float& x:s.obs)require(bool(in>>x),"truncated primitive observation");samples.push_back(s);
        }
        require(!samples.empty()&&in.eof(),"invalid primitive corpus");std::mt19937 rng(seed+400);
        for(int pass=0;pass<5;++pass){std::shuffle(samples.begin(),samples.end(),rng);for(const auto& sample:samples)r.memory.skills.fit(sample);}
    }
    static LanguageReasoner scene(std::istream& in,const std::vector<std::string>& header,
                                  const LanguageReasoner& prior,RevisableGrounding& journal) {
        require(header.size()==4,"invalid scene header");auto r=prior;feed(r,"\n<|endofdialogue|>\n");
        int contexts=std::stoi(header[1]),reports=std::stoi(header[2]);
        require(contexts>=0&&contexts<=4096&&reports>=0&&reports<=2048,"invalid scene size");
        for(int i=0;i<contexts;++i)feed(r,"CONTEXT: "+line(in)+"\n");require(journal.start(r),journal.status);
        std::vector<std::pair<uint64_t,std::string>> retired;
        for(int i=0;i<reports;++i) {
            auto row=fields(line(in));require(row.size()==4,"invalid report row");auto id=std::stoull(row[0]);
            feed(r,"ENVIRONMENT: "+row[3]+"\n");require(journal.append(r,id,row[1])&&journal.consistent,journal.status);
            require(row[2]=="0"||row[2]=="1","invalid report activity");if(row[2]=="0")retired.emplace_back(id,row[1]);
        }
        for(const auto& [id,source]:retired)require(journal.revise_report(r,id,0,"",source,false)&&journal.consistent,journal.status);
        return r;
    }
    static std::shared_ptr<GroundedSession> load(const std::string& manifest,int seed) {
        auto input=open(manifest);std::map<std::string,std::string> paths;
        while(input.peek()!=EOF) {
            auto row=fields(line(input));require(row.size()==2&&!paths.count(row[0]),"invalid path manifest");
            auto path=std::filesystem::path(row[1]);if(path.is_relative())path=std::filesystem::path(manifest).parent_path()/path;
            paths[row[0]]=path.string();
        }
        for(const auto& key:{"micro","bootstrap","incremental","dialogue","observation","references","corrections"})require(paths.count(key),std::string("missing path ")+key);
        auto r=std::make_shared<LanguageReasoner>(seed);primitive(*r,paths.at("micro"),seed);
        BootstrapGrounding bootstrap;require(bootstrap.fit(lessons(paths.at("bootstrap")),r->memory)&&bootstrap.publish(*r),"bootstrap QA fit failed");
        if(seed%2) {
            for(auto& [key,frame]:r->events.frames){std::set<EventGrounding::Rule> swapped;for(auto rule:frame.candidates){rule.relation^=1;swapped.insert(rule);}frame.candidates=std::move(swapped);}
            for(auto& [cue,program]:r->memory.procedures)for(int& op:program)if((op>=4&&op<=7)||op>=10)op^=1;
        }
        IncrementalGrounding incremental,dialogue;
        require(incremental.fit(lessons(paths.at("incremental")),*r)&&incremental.publish(*r),"incremental QA fit failed");
        require(dialogue.fit(lessons(paths.at("dialogue")),*r)&&dialogue.publish(*r),"dialogue QA fit failed");
        r->environment_dialogue=r->grounded_requests=true;r->world_budget=128;
        ObservationLanguageLearner observations;
        require(observations.fit(lessons(paths.at("observation")),*r)&&observations.publish(*r),"observation QA fit failed");
        auto session=std::make_shared<GroundedSession>(r);session->mention_field=seed%2;session->source_field=1-seed%2;
        auto references=open(paths.at("references"));int count=std::stoi(line(references));
        for(int i=0;i<count;++i) {
            auto h=fields(line(references));RevisableGrounding journal;auto reader=scene(references,h,*r,journal);
            auto records=HistoryReferenceGrounding::records(journal,reader,session->mention_field,session->source_field);
            for(int n=0;n<std::stoi(h[3]);++n) {
                auto qa=fields(line(references));require(qa.size()==3,"invalid reference QA");
                require(session->references.teach(reader,journal,records,qa[0],qa[1],qa[2],std::stoull(h[0]),n),session->references.status);
            }
        }
        require(count>0&&references.peek()==EOF,"invalid reference corpus size");
        auto commands=open(paths.at("corrections"));count=std::stoi(line(commands));
        for(int i=0;i<count;++i) {
            auto h=fields(line(commands));RevisableGrounding journal;auto reader=scene(commands,h,*r,journal);auto utterance=line(commands);
            std::vector<CorrectionLanguageGrounding::QA> feedback;
            for(int n=0;n<std::stoi(h[3]);++n){auto qa=fields(line(commands));require(qa.size()==2,"invalid correction QA");feedback.emplace_back(qa[0],qa[1]);}
            auto records=HistoryReferenceGrounding::records(journal,reader,session->mention_field,session->source_field);
            require(session->corrections.teach(reader,journal,session->references,records,utterance,feedback,std::stoull(h[0]),0),session->corrections.status);
        }
        require(count>0&&commands.peek()==EOF,"invalid correction corpus size");
        r->reset_episode();return session;
    }
};
} // namespace laminar::reasoning
