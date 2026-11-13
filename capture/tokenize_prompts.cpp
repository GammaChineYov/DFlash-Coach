#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
using json=nlohmann::ordered_json;
int main(int argc,char **argv) {
    if(argc!=4&&argc!=5) return 1;
    llama_log_set([](ggml_log_level,const char *,void *){},nullptr);
    auto params=llama_model_default_params();
    params.vocab_only=true;
    params.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],params);
    if(!model) return 2;
    auto *vocab=llama_model_get_vocab(model);
    std::ifstream input(argv[2]);std::string line;
    int count=0,invalid=0,maximum=0;const int limit=std::atoi(argv[3]);
    while(std::getline(input,line)) {
        auto row=json::parse(line);
        const std::string prompt=row.at("prompt");
        const int n=-llama_tokenize(vocab,prompt.c_str(),prompt.size(),nullptr,0,false,true);
        maximum=std::max(maximum,n);invalid+=n<=0||n>limit;++count;
        json result={{"id",row.at("id")},{"prompt_tokens",n},{"valid",n>0&&n<=limit}};
        if(argc==5) {
            std::vector<llama_token> ids(n);
            if(llama_tokenize(vocab,prompt.c_str(),prompt.size(),ids.data(),n,false,true)!=n) return 4;
            result["tokens"]=ids;
        }
        std::printf("%s\n",result.dump().c_str());
    }
    std::printf("%s\n",json({{"summary",true},{"rows",count},{"invalid",invalid},{"max_prompt_tokens",maximum}}).dump().c_str());
    llama_model_free(model);
    return invalid?3:0;
}
